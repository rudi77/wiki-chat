import os
import glob
import shutil
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self, parent_dir="./vectordbs"):
        self.parent_dir = parent_dir
        self._ensure_parent_dir()

    def _ensure_parent_dir(self):
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)

    def create_vectordb(self, db_name: str) -> bool:
        db_path = os.path.join(self.parent_dir, db_name)
        if os.path.exists(db_path):
            return False  # Vectordb already exists
        os.makedirs(db_path)
        # Initialize empty Chroma vectorstore
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        Chroma(persist_directory=db_path, embedding_function=embeddings)
        return True

    def list_vectordbs(self) -> list:
        return [name for name in os.listdir(self.parent_dir)
                if os.path.isdir(os.path.join(self.parent_dir, name))]

    def delete_vectordb(self, db_name: str) -> bool:
        db_path = os.path.join(self.parent_dir, db_name)
        if not os.path.exists(db_path):
            return False  # Vectordb does not exist
        shutil.rmtree(db_path)
        return True

    def get_vectorstore(self, db_name: str):
        db_path = os.path.join(self.parent_dir, db_name)
        if not os.path.exists(db_path):
            return None
        embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

    def add_documents(self, db_name: str, documents: list) -> bool:
        vectorstore = self.get_vectorstore(db_name)
        if vectorstore is None:
            return False
        try:
            vectorstore.add_documents(documents=documents)
            vectorstore.persist()
            return True
        except Exception as e:
            print(f"Error adding documents to vectordb '{db_name}': {e}")
            return False

    def list_documents(self, db_name: str) -> list:
        vectorstore = self.get_vectorstore(db_name)
        if vectorstore is None:
            print(f"Vectordb '{db_name}' does not exist.")
            return []

        try:
            # Use Chroma's retriever to fetch metadata
            docs_with_metadata = vectorstore._collection.get(include=['metadatas', 'documents'])
            documents = zip(docs_with_metadata['documents'], docs_with_metadata['metadatas'])

            # Return structured data
            return [
                {
                    "content": doc_content,
                    "metadata": metadata or {}
                }
                for doc_content, metadata in documents
            ]
        except Exception as e:
            print(f"Error listing documents in vectordb '{db_name}': {e}")
            return []


    def delete_document(self, db_name: str, document_id: str) -> bool:
        vectorstore = self.get_vectorstore(db_name)
        if not vectorstore:
            return False
        try:
            vectorstore._collection.delete(ids=[document_id])
            vectorstore.persist()
            return True
        except Exception as e:
            print(f"Error deleting document '{document_id}' from vectordb '{db_name}': {e}")
            return False
