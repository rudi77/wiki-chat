import os
import glob
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.vectorstore = self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        if os.path.exists(os.path.join(self.persist_dir)):
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))
        else:
            # create directory if it does not exist
            os.makedirs(os.path.join(self.persist_dir))

            # Create a new vector store
            return Chroma(
                persist_directory=self.persist_dir,
                embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")))

    def process_documents(self, directory, file_types, splitter_type="Recursive", chunk_size=1000, chunk_overlap=200):
        """
        - Loads and splits new documents from a directory and creates or updates the vector store.
        - Create summary and table of contents for these documents.

        Args:
        - directory (str): The directory path to scan for files.
        - file_types (list): The list of file types to load.
        - splitter_type (str): The type of text splitter to use.
        - chunk_size (int): The size of the text chunks.
        - chunk_overlap (int): The overlap between text chunks.

        Returns:
        - documents (list): The list of processed documents
        """


        documents = self.load_and_split_documents(directory, file_types, splitter_type, chunk_size, chunk_overlap)        
        if documents:    
            self.create_vectorstore(documents)

            # write documents to a file
            with open("documents.txt", "w") as f:
                for doc in documents:
                    f.write(f"{doc.metadata['source']} - {doc.metadata['chunk']}\n{doc.page_content}\n\n")

            return documents
        else:
            return []

    def load_and_split_documents(self, directory, file_types, splitter_type, chunk_size, chunk_overlap):
        all_file_paths = []
        for file_type in file_types:
            file_pattern = f"{directory}/**/*.{file_type}"
            matched = glob.glob(file_pattern, recursive=True)
            all_file_paths.extend(matched)

        text_splitter = self.get_text_splitter(splitter_type, chunk_size, chunk_overlap)
        documents = []
        for file_path in all_file_paths:
            loader = self.get_loader(file_path)
            file_docs = loader.load()
            split_docs = text_splitter.split_documents(file_docs)
            for i, doc in enumerate(split_docs):
                doc.metadata["source"] = file_path
                doc.metadata["chunk"] = i
                documents.append(doc)
        return documents

    def get_loader(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".md":
            return UnstructuredMarkdownLoader(file_path)
        else:
            return TextLoader(file_path)

    def get_text_splitter(self, splitter_type, chunk_size, chunk_overlap):
        if splitter_type == "Markdown":
            return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def create_vectorstore(self, documents):
        if not documents:
            return
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=self.persist_dir)
        self.vectorstore.persist()

    def add_documents(self, pages: list):
        """
        Add documents to the vector store.

        Args:
        - pages (list): The list of pages to add to the vector store. A page is a tuple of (content, metadata).        
        """

        if not self.vectorstore:
            return

        documents = []
        for content, metradata in pages:
            doc = Document(page_content=content, metadata=metradata)
            documents.append(doc)
            
        self.vectorstore.add_documents(documents=documents)
        self.vectorstore.persist()
