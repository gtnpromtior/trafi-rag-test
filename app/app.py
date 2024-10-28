from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
import os
import dotenv
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langfuse.callback import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
import json
import hashlib
from google.oauth2 import service_account
from googleapiclient.discovery import build

dotenv.load_dotenv()

os.environ["GOOGLE_ACCOUNT_FILE"] = os.path.abspath("credentials/credentials.json")

langfuse_handler = CallbackHandler()

handler = CallbackHandler(user_id="gasteac")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

embedding = OpenAIEmbeddings()

def get_drive_files_metadata(folder_id):
    """Get metadata of files in Google Drive folder recursively"""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            os.environ["GOOGLE_ACCOUNT_FILE"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        service = build('drive', 'v3', credentials=credentials)
        files_data = []

        def get_files_in_folder(folder_id):
            query = f"'{folder_id}' in parents and trashed = false"
            page_token = None
            
            while True:
                response = service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, modifiedTime, mimeType)',
                    pageToken=page_token,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True
                ).execute()
                
                for file in response.get('files', []):
                    # Add file to our list if it's not trashed
                    if not file.get('trashed', False):
                        file_data = {
                            'id': file.get('id'),
                            'name': file.get('name'),
                            'modified_time': file.get('modifiedTime'),
                            'mime_type': file.get('mimeType')
                        }
                        files_data.append(file_data)
                        
                        if file.get('mimeType') == 'application/vnd.google-apps.folder':
                            get_files_in_folder(file.get('id'))
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break

        get_files_in_folder(folder_id)
        return files_data
        
    except Exception as e:
        st.error(f"Error getting drive metadata: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def save_files_state(files_data):
    """Save the current state of files"""
    try:
        with open('files_state.json', 'w') as f:
            json.dump(files_data, f)
    except Exception as e:
        st.error(f"Error saving files state: {str(e)}")

def load_files_state():
    """Load the previous state of files"""
    try:
        if os.path.exists('files_state.json'):
            with open('files_state.json', 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading files state: {str(e)}")
        return None

def has_files_changed(folder_id):
    """Check if any files have been modified, added, removed or moved"""
    current_state = get_drive_files_metadata(folder_id)
    previous_state = load_files_state()
    
    if not previous_state or not current_state:
        save_files_state(current_state)
        return True
    
    current_files = {f['id']: f for f in current_state}
    previous_files = {f['id']: f for f in previous_state}
    
    changes_detected = False
    
    for file_id, current_file in current_files.items():
        if file_id not in previous_files:
            st.info(f"New file detected: {current_file['name']}")
            changes_detected = True
        else:
            prev_file = previous_files[file_id]
            if current_file['modified_time'] != prev_file['modified_time']:
                st.info(f"File modified: {current_file['name']}")
                changes_detected = True
    
    for file_id, prev_file in previous_files.items():
        if file_id not in current_files:
            st.info(f"File removed: {prev_file['name']}")
            changes_detected = True
    
    if changes_detected:
        save_files_state(current_state)
        return True
        
    return False

def update_vectorstore_incrementally(vector_store, folder_id, current_state, previous_state):
    """Update vector store incrementally by only processing changed files"""
    try:
        current_files = {f['id']: f for f in current_state}
        previous_files = {f['id']: f for f in previous_state} if previous_state else {}
        
        new_or_modified_ids = []
        removed_ids = []
        
        for file_id, current_file in current_files.items():
            if (file_id not in previous_files or 
                current_file['modified_time'] != previous_files[file_id]['modified_time']):
                new_or_modified_ids.append((file_id, current_file['name']))
        
        for file_id in previous_files:
            if file_id not in current_files:
                removed_ids.append(file_id)
        
        if not new_or_modified_ids and not removed_ids:
            return vector_store
        
        if new_or_modified_ids:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=250,
                chunk_overlap=0,
                model_name="gpt-4o-mini",
            )
            
            conversion_mapping = {
                "application/pdf": PyMuPDFLoader,
                "application/vnd.google-apps.document": GoogleDriveLoader,
                "application/vnd.google-apps.spreadsheet": GoogleDriveLoader,
                "text/plain": TextLoader,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
                "text/csv": CSVLoader,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": CSVLoader,
            }
            
            for file_id, file_name in new_or_modified_ids:
                st.info(f"Processing changes for: {file_name}")
                
                try:
                    loader = GoogleDriveLoader(
                        folder_id=folder_id,
                        template="gdrive-by-name-in-folder",
                        query=file_name,
                        recursive=False,
                        conv_mapping=conversion_mapping,
                    )
                    documents = loader.load()
                    
                    if documents:
                        for doc in documents:
                            doc.metadata['file_id'] = file_id
                            
                        split_docs = text_splitter.split_documents(documents)
                        
                        for doc in split_docs:
                            doc.metadata['file_id'] = file_id
                        
                        try:
                            results = vector_store.similarity_search_with_score(
                                "dummy query",
                                k=1000,
                                filter={"file_id": file_id}
                            )
                            if results:
                                doc_ids_to_delete = [doc[0].metadata.get('doc_id') for doc in results if doc[0].metadata.get('doc_id')]
                                if doc_ids_to_delete:
                                    vector_store.delete(ids=doc_ids_to_delete)
                        except Exception as delete_error:
                            st.warning(f"Could not delete old versions: {str(delete_error)}")
                        
                        for i, doc in enumerate(split_docs):
                            doc.metadata['doc_id'] = f"{file_id}_{i}"
                        
                        vector_store.add_documents(split_docs)
                        
                except Exception as e:
                    st.error(f"Error processing file {file_name}: {str(e)}")
        
        if removed_ids:
            for file_id in removed_ids:
                file_info = previous_files[file_id]
                st.info(f"Removing deleted file from index: {file_info['name']}")
                try:
                    results = vector_store.similarity_search_with_score(
                        "dummy query",
                        k=1000,
                        filter={"file_id": file_id}
                    )
                    if results:
                        doc_ids_to_delete = [doc[0].metadata.get('doc_id') for doc in results if doc[0].metadata.get('doc_id')]
                        if doc_ids_to_delete:
                            vector_store.delete(ids=doc_ids_to_delete)
                except Exception as delete_error:
                    st.warning(f"Could not delete file {file_info['name']}: {str(delete_error)}")
        
        vector_store.save_local("faiss_index")
        return vector_store
        
    except Exception as e:
        st.error(f"Error in incremental update: {str(e)}")
        return vector_store

def get_vectorStore_from_sources():
    try:
        folder_id = "1rpW2q8EqNGz9_GwSuQ448QUBkUIIMxJp"
        
        current_state = get_drive_files_metadata(folder_id)
        previous_state = load_files_state()
        
        if not previous_state or not current_state:
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index")
            
            vector_store = create_full_vectorstore(folder_id)
            if vector_store:
                save_files_state(current_state)
            return vector_store
        
        if os.path.exists("faiss_index"):
            try:
                vector_store = FAISS.load_local(
                    "faiss_index",
                    embedding,
                    allow_dangerous_deserialization=True,
                )
                
                if has_files_changed(folder_id):
                    st.info("Updating index with changes...")
                    vector_store = update_vectorstore_incrementally(
                        vector_store, 
                        folder_id, 
                        current_state, 
                        previous_state
                    )
                    save_files_state(current_state)
                
                return vector_store
                
            except Exception as e:
                st.error(f"Error loading existing index, rebuilding: {str(e)}")
                return create_full_vectorstore(folder_id)
        else:
            return create_full_vectorstore(folder_id)

    except Exception as e:
        st.error(f"Error in vector store creation: {str(e)}")
        return None

def create_full_vectorstore(folder_id):
    """Create a new vector store from all documents"""
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250,
            chunk_overlap=0,
            model_name="gpt-4o-mini",
        )

        conversion_mapping = {
            "application/pdf": PyMuPDFLoader,
            "application/vnd.google-apps.document": GoogleDriveLoader,
            "application/vnd.google-apps.spreadsheet": GoogleDriveLoader,
            "text/plain": TextLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/csv": CSVLoader,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": CSVLoader,
        }
        
        loader = GoogleDriveLoader(
            folder_id=folder_id,
            template="gdrive-by-name-in-folder",
            recursive=True,
            conv_mapping=conversion_mapping,
        )

        documents = loader.load()
        if not documents:
            st.error("No documents were loaded from Google Drive")
            return None

        current_state = get_drive_files_metadata(folder_id)
        file_name_to_id = {f['name']: f['id'] for f in current_state}
        
        for doc in documents:
            file_name = doc.metadata.get('source', '').split('/')[-1]
            if file_name in file_name_to_id:
                doc.metadata['file_id'] = file_name_to_id[file_name]

        split_documents = text_splitter.split_documents(documents)
        if not split_documents:
            st.error("No documents were split successfully")
            return None

        for i, doc in enumerate(split_documents):
            if 'file_id' in doc.metadata:
                doc.metadata['doc_id'] = f"{doc.metadata['file_id']}_{i}"

        vector_store = FAISS.from_documents(split_documents, embedding)
        vector_store.save_local("faiss_index")
        return vector_store

    except Exception as e:
        st.error(f"Error creating full vector store: {str(e)}")
        return None

st.set_page_config(
    page_title="Trafilea RAG",
    page_icon="🚀",
    layout="wide",  
    initial_sidebar_state="expanded",
)

st.header("Ask anything about Trafilea 🤖")
st.toast("Feel free to visit our website: https://www.promtior.ai", icon="🌐")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm here to solve your questions about Trafilea :)"),
    ]

if "vector_store" not in st.session_state:
    with st.spinner("Please wait a moment, I'm loading documents for you 📚"):
        vector_store = get_vectorStore_from_sources()
        if vector_store is None:
            st.error(
                "Failed to initialize the vector store. Please check your credentials and try again."
            )
            st.stop()
        st.session_state.vector_store = vector_store

retriever = st.session_state.vector_store.as_retriever()

def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"]}).to_messages()

retriever_tool = create_retriever_tool(
    retriever, "trafilea_search", "Search for information about Trafilea."
)

tools = [retriever_tool]

system_message = """You are an expert in Metrics from Trafilea, a Tech eCommerce group of Brands.
Answer ONLY based on the documents you have.
Your answers must be long and detailed, using markdown formatting for better readability.

Format your responses following these guidelines:
1. Use **bold** for important terms and metrics
2. Use _italics_ for emphasis
3. Use proper headings with # for main points and ## for subpoints
4. Use bullet points or numbered lists where appropriate
5. For any numerical data or metrics, display them in a clear tabular format using markdown tables
6. Use `code blocks` for any technical terms or specific identifiers
7. Use > for important quotes or highlights

Follow these steps for each query:
1. SEARCH: Use the search tool to find relevant information
2. VERIFY: Check if the retrieved information directly answers the query
3. RESPOND: Provide a well-formatted response using verified information
4. If uncertain, state that clearly

Example formatting:
# Main Response
## Key Findings
- **Metric 1**: Value
- **Metric 2**: Value

| Category | Value | Change |
|----------|--------|--------|
| Sales    | $1000  | +10%   |

> Important highlight or quote

Do not answer any questions that are not directly related to Trafilea."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("placeholder", "{messages}"),
    ]
)

memory = MemorySaver()

model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

app = create_react_agent(
    model, tools, state_modifier=_modify_state_messages, checkpointer=memory
)

config = {
    "configurable": {"thread_id": "test-thread"},
    "recursion_limit": 12,
    "callbacks": [langfuse_handler],
    "metadata": {
        "langfuse_user_id": "gasteac",
    },
}

def get_response(user_input):
    messages = [
        ("human", msg.content) if isinstance(msg, HumanMessage) else ("ai", msg.content)
        for msg in st.session_state.chat_history
    ]
    messages.append(("human", user_input))

    response = app.invoke({"messages": messages}, config)
    return response["messages"][-1].content

user_query = st.chat_input("Write your questions here :)")

if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.extend(
        [HumanMessage(content=user_query), AIMessage(content=response)]
    )

for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "You"):
        if isinstance(message, AIMessage):
            st.markdown(message.content)  # Use st.markdown instead of st.write
        else:
            st.write(message.content)

with st.sidebar:
    st.markdown("### 🚀 Trafilea RAG")
    st.markdown(
        """
    ### About This Assistant:
    - RAG Agents
    - OpenAI GPT-4
    - LangChain and LangGraph
    """
    )

    st.markdown(
        """
        <style>
        .visit-button {
            display: inline-block;
            padding: 8px 20px;
            background-color: #FF4B4B;
            color: white !important;
            text-decoration: none;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            width: 100%;
            transition: background-color 0.3s ease;
        }
        .visit-button:hover {
            background-color: #FF6B6B;
            text-decoration: none;
        }
        </style>
        <a href="https://www.trafilea.com" target="_blank" class="visit-button">🌐 Visit Trafilea</a>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    ### Tips for Better Results
    - Be specific in your questions
    - Ask one thing at a time
    - Provide context
    """
    )
    st.markdown("---")
    st.caption("© 2024 Promtior. All rights reserved.")

