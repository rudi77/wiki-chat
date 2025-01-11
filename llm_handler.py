import os
from langchain_community.chat_models import openai
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self, model="gpt-4o-mini", temperature=0.7):
        self.model = model
        self.temperature = temperature
        self.llm = self.initialize_llm()

    def initialize_llm(self):
        return openai.ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=self.model,
            temperature=self.temperature
        )

    def create_retrieval_chain(self, vectorstore):
        if not vectorstore:
            return None
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert support assistant. Use the given context to answer the question."),
            ("human", "Context: {context}\nQuestion: {input}")
        ])
        combine_documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        retriever = vectorstore.as_retriever()
        return create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_documents_chain
        )

    def send_query(self, system_prompt: str, user_prompt: str):
        response = self.llm([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content.strip()