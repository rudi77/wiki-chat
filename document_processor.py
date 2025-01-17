import os
import glob
import json
import uuid  # Moved import to the top for better practice
from concurrent.futures import ThreadPoolExecutor, as_completed
from vector_store import VectorStoreManager
from llm_handler import LLMHandler
from langchain.schema import Document

# Add the missing imports below
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

class DocumentProcessor:
    """
    The DocumentProcessor class takes documents of any kind and processes 
    them so that they can be stored in a vector store.
    """
    def __init__(self, vectorstore_manager: VectorStoreManager, llm: LLMHandler):
        self.vectorstore_manager = vectorstore_manager
        self.llm = llm

    def process_documents(self, db_name: str, directory: str, file_types: list, splitter_type="Recursive", chunk_size=2000, chunk_overlap=200, progress_callback=None):
        """
        Reads documents recursively from a directory and processes them for storage in the vector store.
        - Create a summary of the content of each document.
        - Create a table of contents based on the summaries.
        - Store the content, summary, and table of contents in the vector store.
        - Content will be chunked and vectorized for similarity search.

        Args:
        - db_name (str): The name of the vectordb to add documents to.
        - directory (str): The directory path to scan for files.
        - file_types (list): The list of file types to be supported.
        - splitter_type (str): The type of text splitter to use.
        - chunk_size (int): The size of the text chunks.
        - chunk_overlap (int): The overlap between text chunks.
        """

        # Check if the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found.")
        
        files = []

        # Iterate through the files in the directory recursively. Ignore directories starting with "."
        for file_type in file_types:
            # Do not include directories starting with "."
            files.extend(glob.glob(os.path.join(directory, f"**/*.{file_type}"), recursive=True))

        if not files:
            print("No files found matching the specified file types.")
            return

        # Add summaries of the files to the vector store
        summaries = self.add_file_summaries(files, read_from_file=False, db_name=db_name, progress_callback=progress_callback)

        # Add table of contents to the vector store
        self.add_master_toc(db_name, self._combine_summaries(summaries))
        
        # Load and split documents
        documents = self.load_and_split_documents(directory, file_types, splitter_type, chunk_size, chunk_overlap)
        
        if documents:
            # Prepare documents for addition to the vector store
            # Convert content and metadata into Document objects
            documents = [
                self.create_document(page_content=doc.page_content, metadata={
                    "source": doc.metadata.get("source", "N/A"),
                    "chunk": doc.metadata.get("chunk", 0),
                    "id": self.generate_doc_id()
                })
                for doc in documents
            ]
            
            # Add documents to the vector store
            success = self.vectorstore_manager.add_documents(db_name, documents=documents)
            if success:
                print("Documents added successfully to the vector store.")
            else:
                print("Failed to add documents to the vector store.")
            
            # Optionally, write documents to a file for record-keeping
            filename = db_name + "_documents.txt" if db_name else "documents.txt"
            with open(db_name, "w", encoding='utf-8') as f:
                for doc in documents:
                    f.write(f"{doc.metadata.get('source', 'Unknown Source')} - Chunk {doc.metadata.get('chunk', 'N/A')}\n{doc.page_content}\n\n")
        else:
            print("No documents to add to the vector store.")

    def add_file_summaries(self, files, read_from_file=False, db_name: str = "", progress_callback=None):
        """
        Summarizes the content of the files and adds the summaries to the vector store.

        Args:
        - files (list): A list of file paths to summarize.
        - read_from_file (bool): Whether to read existing summaries from a file.
        - db_name (str): The name of the vectordb to add the summaries to.

        Returns:
        - summaries (dict): A dictionary of file summaries.
        """
        summaries = {}

        # Load existing summaries from a file if requested
        if read_from_file and os.path.exists("summaries.json"):
            with open("summaries.json", "r", encoding='utf-8') as f:
                summaries = json.load(f)
        else:
            def process_file(file):
                """Helper function to process a single file and generate a summary."""
                try:
                    with open(file, "r", encoding='utf-8') as f:
                        file_text = f.read()
                        user_prompt = f"Provide a concise summary of the file '{file}':\n{file_text}"
                        system_prompt = "You are a helpful assistant for summarizing files."
                        summary = self.llm.send_query(system_prompt, user_prompt)
                        return file, {
                            "summary": summary,
                            "content": file_text,
                            "file_name": os.path.basename(file)
                        }
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
                    return file, None

            total_files = len(files)


            # Use ThreadPoolExecutor to process files in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_file = {executor.submit(process_file, file): file for file in files}

                for idx, future in enumerate(as_completed(future_to_file), start=1):
                    file, result = future.result()
                    if result:
                        summaries[file] = result

                     # Invoke the progress callback with the current state
                    if progress_callback:
                        progress_callback(idx, total_files, file)

            # Save summaries to a JSON file
            summaries_json = json.dumps(summaries, indent=4)
            filename = db_name + "_summaries.json" if db_name else "summaries.json"
            with open(filename, "w", encoding='utf-8') as f:
                f.write(summaries_json)

        # Add the summaries to the vector store
        summary_list = [
            self.create_document(page_content=summary["summary"], metadata={
                "source": summary['file_name'],
                "description": "Summary of file",
                "type": "summary",
                "id": self.generate_doc_id()
            })
            for summary in summaries.values()
        ]

        if db_name:  # Ensure db_name is provided
            self.vectorstore_manager.add_documents(db_name=db_name, documents=summary_list)

        return summaries

    def _combine_summaries(self, summaries: dict):
        """
        Combines the summaries of the files into a single table of contents.
        A single summary has the following format:
        {
            "summary" : summary,
            "content" : file_text,
            "file_name" : os.path.basename(file)
        }
        """

        combined_summaries = ""
        for summary in summaries.values():
            combined_summaries += f"FILE: {summary['file_name']}\nSUMMARY: {summary['summary']}\n\n"

        print(combined_summaries)

        return combined_summaries

    def add_master_toc(self, db_name: str, combined_summaries: str, read_from_file=False):
        """
        Creates a table of contents based on the summaries of the files.
        """
        if read_from_file and os.path.exists("toc.md"):
            with open("toc.md", "r", encoding='utf-8') as f:
                toc = f.read()
        else:
            user_content = f"You are an expert content organizer. Based on the following summaries, create a Table of Contents in Markdown:\n\n{combined_summaries}"
            system_prompt = "You are an expert content organizer."
            toc = self.llm.send_query(system_prompt, user_content)
            # Write the table of contents to a toc.md file

            filename = db_name + "_toc.md" if db_name else "toc.md"
            with open(filename, "w", encoding='utf-8') as f:
                f.write(toc)

        # Add the table of contents to the vector store
        self.vectorstore_manager.add_documents(db_name, [
            self.create_document(page_content=toc, metadata={
                "description": "Table of Contents",
                "type": "toc",
                "id": self.generate_doc_id()
            })
        ])            

    def load_and_split_documents(self, directory, file_types, splitter_type, chunk_size, chunk_overlap):
        all_file_paths = []
        for file_type in file_types:
            file_pattern = os.path.join(directory, f"**/*.{file_type}")
            matched = glob.glob(file_pattern, recursive=True)
            all_file_paths.extend(matched)

        text_splitter = self.get_text_splitter(splitter_type, chunk_size, chunk_overlap)
        documents = []
        for file_path in all_file_paths:
            try:
                loader = self.get_loader(file_path)
                file_docs = loader.load()
                split_docs = text_splitter.split_documents(file_docs)
                for i, doc in enumerate(split_docs):
                    doc.metadata["source"] = file_path
                    doc.metadata["chunk"] = i
                    documents.append(doc)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
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

    def generate_doc_id(self):
        """
        Generates a unique document ID.
        """
        return str(uuid.uuid4())

    def create_document(self, page_content: str, metadata: dict) -> Document:
        """
        Create a Document object from the given content and metadata.
        
        Args:
            page_content (str): The content of the document.
            metadata (dict): Metadata associated with the document.
        
        Returns:
            Document: The created Document object.
        """
        return Document(page_content=page_content, metadata=metadata)