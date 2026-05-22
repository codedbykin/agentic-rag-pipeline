import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
import requests

load_dotenv()

app = FastAPI(title="Agentic RAG API")

class Question(BaseModel):
    question: str
    include_weather: bool = False
    city: str = "Dublin"

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

@app.get("/")
def home():
    return {"message": "Agentic RAG API is running!"}

@app.post("/ask")
def ask(body: Question):
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )

    doc_context = search_documents(body.question)
    
    weather_info = ""
    if body.include_weather:
        weather_info = f"\nCurrent weather in {body.city}: {get_weather(body.city)}"

    messages = [
        SystemMessage(content="You are a helpful assistant. Answer based on the document context provided."),
        HumanMessage(content=f"""
Document context:
{doc_context}
{weather_info}

Question: {body.question}
""")
    ]

    response = llm.invoke(messages)
    return {"answer": response.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)