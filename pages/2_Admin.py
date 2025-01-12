# pages/2_Admin.py
import streamlit as st
from llm_handler import LLMHandler
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    st.set_page_config(page_title="Admin - Wiki Q&A Chatbot", layout="wide")
    st.title("🛠️ Admin & Management")

    # Sidebar Settings
    st.sidebar.header("⚙️ Admin Settings")
    model_choice = st.sidebar.selectbox("Select model:", ["gpt-4o-mini", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    persist_directory = st.sidebar.text_input("Persist Directory for Vector Store:", value="./chroma_db")
    directory = st.sidebar.text_input("Directory path to scan for files:", value="", key="admin_directory_input")
    file_types_selected = st.sidebar.multiselect("Select file types to load:", ["md", "py", "cs", "txt"], default=["md"])
    splitter_type = st.sidebar.selectbox("Splitter Type", ["Recursive", "Markdown"], index=0)
    chunk_size = st.sidebar.number_input("Chunk size:", min_value=100, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk overlap:", min_value=0, value=200, step=50)

    # Initialize Vector Store
    if 'vsm' not in st.session_state:
        vsm = VectorStoreManager(persist_dir=persist_directory)
        st.session_state.vsm = vsm

    if 'retrieval_chain' not in st.session_state:
        llm_handler = LLMHandler(model=model_choice, temperature=temperature)
        st.session_state.retrieval_chain = llm_handler.create_retrieval_chain(st.session_state.vectorstore)
        st.session_state.llm_handler = llm_handler

    # Initialize Session State Variable
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

    # Process Documents Button
    if st.button("📂 Process Documents"):
        if not directory:
            st.warning("⚠️ Please enter a directory path.")
        else:
            # start a spinner
            with st.spinner("Processing documents..."):
                doc_processor = DocumentProcessor(vectorstore_manager=st.session_state.vsm, llm=st.session_state.llm_handler)
                doc_processor.process_documents(directory, file_types_selected)
                st.write("✅ Documents processed successfully.")

main()
