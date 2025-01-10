import streamlit as st
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    st.set_page_config(page_title="Wiki Q&A Chatbot", layout="wide")
    st.sidebar.header("LLM Settings")
    model_choice = st.sidebar.selectbox("Select model:", ["gpt-4o-mini", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.sidebar.header("File Loading Settings")
    directory = st.sidebar.text_input("Directory path to scan for files:", value="", key="directory_input")
    file_types_selected = st.sidebar.multiselect("Select file types to load:", ["md", "py", "cs", "txt"], default=["md"])
    chunk_size = st.sidebar.number_input("Chunk size:", min_value=100, value=1000, step=100)
    chunk_overlap = st.sidebar.number_input("Chunk overlap:", min_value=0, value=200, step=50)
    splitter_type = st.sidebar.selectbox("Splitter Type", ["Recursive", "Markdown"], index=0)
    persist_directory = st.sidebar.text_input("Persist Directory for Vector Store:", value="./chroma_db")
    clear_button = st.sidebar.button("Clear Chat and Reload")

    if clear_button:
        for key in ["messages", "documents_processed", "file_summaries", "table_of_contents"]:
            if key in st.session_state:
                del st.session_state[key]
        vsm = VectorStoreManager(persist_dir=persist_directory)
        st.session_state.vectorstore = vsm.load_vectorstore()
        st.rerun()

    vsm = VectorStoreManager(persist_dir=persist_directory)
    llm_handler = LLMHandler(model=model_choice, temperature=temperature)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "file_summaries" not in st.session_state:
        st.session_state.file_summaries = {}
    if "table_of_contents" not in st.session_state:
        st.session_state.table_of_contents = ""

    tab_chat, tab_manage = st.tabs(["Chat", "Manage Summaries & ToC"])

    with tab_chat:
        if "retrieval_chain" not in st.session_state:
            if vsm.vectorstore:
                retrieval_chain = llm_handler.create_retrieval_chain(vsm.vectorstore)
                st.session_state.retrieval_chain = retrieval_chain
            else:
                st.warning("No vector store available. Please process documents and generate summaries.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer:**\n{message['content']}")
            elif message["role"] == "retriever":
                with st.chat_message("retriever"):
                    st.markdown(f"**Context from RAG:**\n{message['content']}")
            else:
                with st.chat_message("user"):
                    st.markdown(message["content"])

        user_input = st.chat_input("Ask a question about your files:")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            if st.session_state.vectorstore:
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 20})
                context_docs = retriever.get_relevant_documents(user_input)
                formatted_context = "\n\n".join([doc.page_content for doc in context_docs])
                st.session_state.messages.append({"role": "retriever", "content": formatted_context})

            if st.session_state.retrieval_chain:
                with st.spinner("Generating answer from LLM..."):
                    response = st.session_state.retrieval_chain.invoke({"input": user_input})
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                    with st.chat_message("assistant"):
                        st.markdown(f"**Answer:**\n{response['answer']}")     

    with tab_manage:
        if st.button("Process Documents"):
            if not directory:
                st.warning("Please enter a directory path.")
            else:
                documents = vsm.process_documents(directory, file_types_selected, splitter_type, chunk_size, chunk_overlap)
                if documents:
                    st.success(f"Processed {len(documents)} documents.")
                    st.session_state.documents_processed = True
                else:
                    st.warning("No documents to process.")

        if st.session_state.get("documents_processed", False):
            if st.button("Generate Summaries"):
                # Logic to generate summaries
                pass
            if st.button("Generate Table of Contents"):
                # Logic to generate table of contents
                pass
        else:
            st.info("Please process documents first.")

