import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_documents(docs_folder="docs"):
    print("Loading documents...")
    
    all_docs = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(docs_folder, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"Loaded: {filename}")
    
    if not all_docs:
        print("No PDF files found in docs folder!")
        return
    
    print(f"Total pages loaded: {len(all_docs)}")
    
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")
    
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Done! Documents stored in ChromaDB.")

if __name__ == "__main__":
    ingest_documents()