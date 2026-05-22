# 🤖 Agentic RAG Pipeline

An autonomous AI agent that retrieves information from documents and combines it with real-time external data — built with LangChain, Azure OpenAI GPT-4o, and ChromaDB.

## 🚀 Features
- 📄 **Document Ingestion** — Automatically reads and indexes PDF documents
- 🔍 **Semantic Search** — Finds relevant information using vector embeddings
- 🤖 **Agentic AI** — Autonomously decides which tools to use
- 🌐 **External API Integration** — Fetches live data (weather, etc.)
- ⚡ **REST API** — Production-ready FastAPI backend
- 📊 **Auto Documentation** — Swagger UI included

## 🛠️ Tech Stack
| Technology | Purpose |
|------------|---------|
| Python | Core language |
| LangChain | AI agent framework |
| Azure OpenAI GPT-4o | Language model |
| ChromaDB | Vector database |
| FastAPI | REST API |
| Uvicorn | ASGI server |

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/codedbykin/agentic-rag-pipeline.git
cd agentic-rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Add your Azure OpenAI credentials
```

### 5. Ingest documents
```bash
python rag/ingest.py
```

### 6. Run the API
```bash
python app.py
```

## 📡 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/ask` | Ask a question |

## 👩‍💻 Author
**Kinjal Solanki**
Cloud Computing | AI Engineering
📍 Dublin, Ireland
