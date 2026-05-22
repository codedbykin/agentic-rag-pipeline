import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
import requests

load_dotenv()

def search_documents(query):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    docs = vectorstore.as_retriever().invoke(query)
    return "\n".join([d.page_content for d in docs])

def get_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url)
    return response.text

def run_agent(question):
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    doc_context = search_documents(question)
    weather = get_weather("Dublin")

    messages = [
        SystemMessage(content="You are a helpful assistant. Use the document context and weather info to answer."),
        HumanMessage(content=f"""
Document context:
{doc_context}

Current weather in Dublin: {weather}

Question: {question}
""")
    ]

    print(f"\nQuestion: {question}")
    print("Thinking...")
    response = llm.invoke(messages)
    print(f"\nAnswer: {response.content}")

if __name__ == "__main__":
    run_agent("What does the document say about data insights? Also what is the weather in Dublin?")