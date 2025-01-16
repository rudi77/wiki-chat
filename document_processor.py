import os
import glob
import json
from vector_store import VectorStoreManager
from llm_handler import LLMHandler


class DocumentProcessor:
    """
    The DocumentProcessor class takes documents of any kind and processes 
    them so that they can be stored in a vector store.
    """
    def __init__(self, vectorstore_manager: VectorStoreManager, llm: LLMHandler):
        self.vectorstore_manager = vectorstore_manager
        self.llm = llm

    def process_documents(self, directory, file_types):
        """
        Reads documents recursively from a directory and processes them for storage in the vector store.
        - Create a summary of the content of each document.
        - Create a table of contents based on the summaries.
        - Store the content, summary, and table of contents in the vector store.
        - Content will be chunked and vectorized for similarity search.

        Args:
        - directory (str): The directory path to scan for files.
        - file_types (list): The list of file types to be supported.
        """

        # check if the directory exists
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} not found.")
        
        files = []

        # Iterate through the files in the directory recursively. Ignore directories starting with "."
        for file_type in file_types:
            # do not include directories starting with "."
            files.extend(glob.glob(f"{directory}/**/*.{file_type}", recursive=True))
    

        # Add summaries of the files to the vector store
        summaries = self.add_file_summaries(files, read_from_file=True)

        # Add table of contents to the vector store
        self.add_master_toc(self._combine_summaries(summaries))
        
        # Add the content of the files to the vector store
        self.vectorstore_manager.process_documents(directory, file_types)

    def add_file_summaries(self, files, read_from_file=False):
        summaries = {}

        if read_from_file:
            with open("summaries.json", "r") as f:
                summaries = json.load(f)
        else:
            for file in files:
                with open(file, "r") as f:
                    file_text = f.read()
                    user_prompt = f"Provide a concise summary of the file '{file}':\n{file_text}"
                    system_prompt = "You are a helpful assistant for summarizing files."
                    summary = self.llm.send_query(system_prompt, user_prompt)
                    summaries[file] = {
                        "summary" : summary,
                        "content" : file_text,
                        "file_name" : os.path.basename(file)
                    }
                        
            # create a json object from the summaries dictionary
            summaries_json = json.dumps(summaries, indent=4)

            # write the summaries to a summaries.json file
            with open("summaries.json", "w") as f:
                f.write(summaries_json)
        
        # Add the documents to the vector store
        summary_list = [
            (summary["summary"], {"source": summary['file_name'], "description": "Summary of file", "type": "summary"})
            for summary in summaries.values()]

        self.vectorstore_manager.add_documents(summary_list)
        
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
            combined_summaries += f"FILE: {summary['file_name']}\nSUMMARY: {summary['summary']}\n"
            combined_summaries += "\n\n"

        print(combined_summaries)

        return combined_summaries

    def add_master_toc(self, combined_summaries, read_from_file=False):
        """
        Creates a table of contents based on the summaries of the files.
        
        """
        if read_from_file:
            with open("toc.md", "r") as f:
                toc = f.read()
        else:
            user_content = f"You are an expert content organizer. Based on the following summaries, create a Table of Contents in Markdown:\n\n{combined_summaries}"
            system_prompt = "You are an expert content organizer."
            toc = self.llm.send_query(system_prompt, user_content)
            # write the table of contents to a toc.md file
            with open("toc.md", "w") as f:
                f.write(toc)

        # Add the table of contents to the vector store
        self.vectorstore_manager.add_documents([(toc, {"description": "Table of Contents", "type": "toc"})])            



        
