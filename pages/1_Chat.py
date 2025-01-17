# pages/1_Chat.py
import streamlit as st
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def initialize_vectorstore(vsm, selected_db):
    """Initialize the vectorstore for the selected database."""
    if "vectorstore" not in st.session_state or st.session_state.get("selected_db") != selected_db:
        vectorstore = vsm.get_vectorstore(selected_db)
        if vectorstore is None:
            st.error(f"Failed to load VectorDB '{selected_db}'.")
            return None
        st.session_state.vectorstore = vectorstore
        st.session_state.selected_db = selected_db
    return st.session_state.vectorstore


def initialize_retrieval_chain(model_choice, temperature):
    """Initialize the LLM retrieval chain."""
    if "retrieval_chain" not in st.session_state:
        llm_handler = LLMHandler(model=model_choice, temperature=temperature)
        retrieval_chain = llm_handler.create_retrieval_chain(st.session_state.vectorstore)
        st.session_state.retrieval_chain = retrieval_chain
        st.session_state.llm_handler = llm_handler
    return st.session_state.retrieval_chain


def reset_chat():
    """Reset the chat session state."""
    keys_to_clear = [
        "messages",
        "documents_processed",
        "file_summaries",
        "table_of_contents",
        "retrieval_chain",
        "llm_handler",
        "context",
        "vectorstore",
        "selected_db",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def main():
    st.set_page_config(page_title="Chat - Wiki Q&A Chatbot", layout="wide")
    st.title("💬 Chat Interface")

    # Sidebar Settings
    st.sidebar.header("⚙️ LLM Settings")
    model_choice = st.sidebar.selectbox("Select model:", ["gpt-4o-mini", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.sidebar.header("📂 Vector Database Selection")
    persist_directory = st.sidebar.text_input("Persist Directory for Vector Store:", value="./chroma_db")
    vsm = VectorStoreManager(parent_dir=persist_directory)
    available_dbs = vsm.list_vectordbs()  # List available vector databases
    selected_db = st.sidebar.selectbox("Select a VectorDB to Chat With:", available_dbs)

    clear_button = st.sidebar.button("🧹 Clear Chat and Reload", on_click=reset_chat)

    # Initialize Vector Store and Retrieval Chain
    vectorstore = initialize_vectorstore(vsm, selected_db)
    if not vectorstore:
        return

    retrieval_chain = initialize_retrieval_chain(model_choice, temperature)
    if not retrieval_chain:
        return

    # Initialize Session State Variables
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    user_input = st.chat_input("Ask a question about your selected VectorDB:")
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process Input and Generate Response
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 50})
        context_docs = retriever.get_relevant_documents(user_input)

        # Log context documents (optional)
        for doc in context_docs:
            logging.info(f"Document Metadata: {doc.metadata}")
            logging.info(f"Document Content: {doc.page_content}")

        formatted_context = "\n\n".join([doc.page_content for doc in context_docs])
        st.session_state.context = formatted_context

        with st.spinner("🧠 Generating answer from LLM..."):
            response = st.session_state.retrieval_chain.invoke({"input": user_input})
            assistant_response = response["answer"]

            # Append assistant message
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)


main()
