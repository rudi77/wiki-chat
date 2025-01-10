# RAG Application: Wiki Q&A Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) application that allows users to interact with documents stored in a specified directory. The application leverages Streamlit for the user interface, LangChain for document processing and retrieval, and OpenAI's GPT models for generating responses.

## Features

- **Document Processing**: Load and process documents from a specified directory.
- **Vector Store Management**: Create and manage a vector store using Chroma and OpenAI embeddings.
- **Chat Interface**: Interact with the documents using a chat interface powered by OpenAI's GPT models.
- **Document Summarization**: Generate summaries for processed documents.
- **Table of Contents Generation**: Create a table of contents based on the summaries of the documents.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rag-application.git
   cd rag-application
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Run the application**:
   ```bash
   poetry run streamlit run ui.py
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://localhost:8501`.

3. **Configure settings**:
   - **LLM Settings**: Select the model and set the temperature.
   - **File Loading Settings**: Specify the directory path, file types, chunk size, chunk overlap, and splitter type.
   - **Persist Directory**: Set the directory for storing the vector store.

4. **Process Documents**:
   - Click the "Process Documents" button to load and process documents from the specified directory.

5. **Generate Summaries and Table of Contents**:
   - After processing documents, you can generate summaries and a table of contents using the respective buttons.

6. **Chat with the Documents**:
   - Use the chat interface to ask questions about the processed documents. The application will retrieve relevant context and generate answers using the selected GPT model.

## File Structure

- `ui.py`: The main Streamlit application file that handles the user interface and interaction.
- `llm_handler.py`: Handles interactions with the OpenAI GPT models, including creating retrieval chains and generating summaries.
- `vector_store.py`: Manages the loading, processing, and storage of documents in a vector store using Chroma and OpenAI embeddings.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://www.langchain.com/) for document processing and retrieval.
- [OpenAI](https://openai.com/) for the GPT models and embeddings.

---

Feel free to customize this README to better fit your project's needs.