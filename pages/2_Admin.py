import streamlit as st
from llm_handler import LLMHandler
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    st.set_page_config(page_title="Admin - Wiki Q&A Chatbot", layout="wide")
    st.title("🛠️ Admin & Management")

    # Initialize VectorStoreManager
    if 'vectorstore_manager' not in st.session_state:
        st.session_state.vectorstore_manager = VectorStoreManager()

    vectorstore_manager = st.session_state.vectorstore_manager

    # Initialize LLMHandler
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler()

    llm_handler = st.session_state.llm_handler

    # Initialize DocumentProcessor
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor(vectorstore_manager, llm_handler)

    document_processor = st.session_state.document_processor

    # Sidebar Settings
    st.sidebar.header("⚙️ Admin Settings")
    model_choice = st.sidebar.selectbox("Select model:", ["gpt-4o-mini", "gpt-4"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    parent_dir = st.sidebar.text_input("Parent Directory for Vector Stores:", value="./chroma_db")

    # Update VectorStoreManager if parent_dir changes
    if vectorstore_manager.parent_dir != parent_dir:
        st.session_state.vectorstore_manager = VectorStoreManager(parent_dir=parent_dir)
        st.session_state.document_processor.vectorstore_manager = st.session_state.vectorstore_manager
        vectorstore_manager = st.session_state.vectorstore_manager

    # Update LLMHandler settings
    if llm_handler.model != model_choice or llm_handler.temperature != temperature:
        st.session_state.llm_handler = LLMHandler(model=model_choice, temperature=temperature)
        st.session_state.document_processor.llm = st.session_state.llm_handler
        llm_handler = st.session_state.llm_handler

    # Tabs for different functionalities
    tabs = st.tabs(["Create New Vectordb", "List Existing Vectordbs", "Manage Vectordb"])

    with tabs[0]:
        st.subheader("Create a New Vector Database")
        new_db_name = st.text_input("Enter new vectordb name:")
        if st.button("Create Vectordb"):
            if not new_db_name.strip():
                st.warning("⚠️ Please enter a valid vectordb name.")
            else:
                success = vectorstore_manager.create_vectordb(new_db_name.strip())
                if success:
                    st.success(f"✅ Vectordb '{new_db_name}' created successfully.")
                else:
                    st.error(f"❌ Vectordb '{new_db_name}' already exists.")

    with tabs[1]:
        st.subheader("Existing Vector Databases")
        existing_dbs = vectorstore_manager.list_vectordbs()
        if existing_dbs:
            for db in existing_dbs:
                st.write(f"- {db}")
        else:
            st.info("No vectordbs found.")

    with tabs[2]:
        st.subheader("Manage a Vector Database")

        existing_dbs = vectorstore_manager.list_vectordbs()
        if not existing_dbs:
            st.info("No vectordbs available to manage.")
            return

        selected_db = st.selectbox("Select a vectordb to manage:", existing_dbs)

        if selected_db:
            st.markdown(f"### Managing Vectordb: **{selected_db}**")

            manage_tabs = st.tabs(["Add Documents", "List Documents", "Delete Documents", "Delete Vectordb"])

            with manage_tabs[0]:
                st.subheader("Add Documents to Vectordb")
                directory = st.text_input("Directory path to scan for files:", value="", key="manage_add_dir")
                file_types_selected = st.multiselect("Select file types to load:", ["md", "py", "cs", "txt", "tf", "tfvars", "yaml", "yml", "json"], default=["md"])
                splitter_type = st.selectbox("Splitter Type", ["Recursive", "Markdown"], index=0)
                chunk_size = st.number_input("Chunk size:", min_value=100, value=1000, step=100)
                chunk_overlap = st.number_input("Chunk overlap:", min_value=0, value=200, step=50)
                
                if st.button("📂 Add Documents"):
                    if not directory.strip():
                        st.warning("⚠️ Please enter a directory path.")
                    elif not os.path.exists(directory):
                        st.error("❌ The specified directory does not exist.")
                    else:
                        with st.spinner("Processing and adding documents..."):
                            try:
                                document_processor.process_documents(
                                    db_name=selected_db,
                                    directory=directory.strip(),
                                    file_types=file_types_selected,
                                    splitter_type=splitter_type,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    progress_callback=lambda current, total, file: st.text(f"Processing {current}/{total}: {file}")
                                )
                                st.success("✅ Documents added successfully.")
                            except Exception as e:
                                st.error(f"❌ Error processing documents: {e}")


            with manage_tabs[1]:
                st.subheader("List Documents in Vectordb")
                documents = vectorstore_manager.list_documents(selected_db)

                if documents:
                    # Iterate over each document
                    for i, doc in enumerate(documents, start=1):
                        # Display the content as is
                        st.markdown(f"### Document {i}")
                        st.markdown(f"**Content:**\n\n{doc['content']}")

                        # Display metadata as a numbered list
                        if doc.get('metadata', {}):
                            st.markdown("**Metadata:**")
                            metadata_lines = [
                                f"{j+1}. **{key}:** {value}"
                                for j, (key, value) in enumerate(doc['metadata'].items())
                            ]
                            st.markdown("\n".join(metadata_lines))
                        else:
                            st.markdown("_No metadata available._")

                        st.markdown("---")  # Separator between documents
                else:
                    st.info("No documents found in this vectordb.")


            # with manage_tabs[2]:
            #     st.subheader("Delete Documents from Vectordb")
            #     documents = vectorstore_manager.list_documents(selected_db)
            #     if documents:
            #         doc_options = [f"{doc['id']} - {doc['source']}" for doc in documents]
            #         selected_docs = st.multiselect(
            #             "Select documents to delete:",
            #             options=doc_options,
            #             format_func=lambda x: x
            #         )
            #         if st.button("🗑️ Delete Selected Documents"):
            #             if not selected_docs:
            #                 st.warning("⚠️ Please select at least one document to delete.")
            #             else:
            #                 deleted = 0
            #                 for doc in selected_docs:
            #                     doc_id = doc.split(" - ")[0]
            #                     success = vectorstore_manager.delete_document(selected_db, doc_id)
            #                     if success:
            #                         deleted += 1
            #                 st.success(f"✅ Deleted {deleted} document(s) successfully.")
            #     else:
            #         st.info("No documents available to delete.")

            with manage_tabs[3]:
                st.subheader("Delete Vectordb")
                if st.button("🗑️ Delete Vectordb"):
                    confirm = st.checkbox(f"Are you sure you want to delete vectordb '{selected_db}'? This action cannot be undone.")
                    if confirm:
                        success = vectorstore_manager.delete_vectordb(selected_db)
                        if success:
                            st.success(f"✅ Vectordb '{selected_db}' deleted successfully.")
                            # Optionally, refresh the selection
                            st.experimental_rerun()
                        else:
                            st.error(f"❌ Failed to delete vectordb '{selected_db}'.")
                    else:
                        st.warning("⚠️ Deletion cancelled.")


main()
