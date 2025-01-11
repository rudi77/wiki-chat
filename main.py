# main.py
import streamlit as st

def main():
    st.set_page_config(page_title="Wiki Q&A Chatbot", layout="wide")
    st.title("📚 Wiki Q&A Chatbot")
    st.write("""
        Welcome to the **Wiki Q&A Chatbot** application!

        Use the sidebar to navigate between the **Chat** interface and the **Admin** management panel.

        - **Chat**: Interact with the chatbot to ask questions about your files.
        - **Admin**: Manage documents, generate summaries, and create a table of contents.

        Select a page from the sidebar to get started.
    """)

if __name__ == "__main__":
    main()