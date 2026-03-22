# Mini-RAG System with LangChain + FAISS + Gemini 2.5 Flash

A production-style Retrieval-Augmented Generation pipeline over multi-source documents: PDF, CSV, and live web content. This project demonstrates a full pipeline from document ingestion to a RESTful API.

## Key Features
- **Multi-source ingestion:** PDF via `PyPDFLoader`, CSV via `pandas`, web via `requests` + `BeautifulSoup`
- **Chunking with overlap:** `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
- **Dense embeddings:** `HuggingFaceEmbeddings` — local, no API cost
- **Vector similarity search:** `FAISS.similarity_search_with_score()`
- **Source-level filtering:** `filter={"source": ...}` on retrieval
- **Prompt engineering:** `ChatPromptTemplate` — grounded, hallucination-guarded
- **Modern LLM chaining:** LCEL pipe syntax: `prompt | llm | StrOutputParser()`
- **REST API:** FastAPI with Pydantic validation and structured responses

## Tech Stack
| Layer | Technology |
|---|---|
| Document loading | LangChain `PyPDFLoader`, `Document`, `requests` + `BeautifulSoup` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap) |
| Embeddings | `all-MiniLM-L6-v2` via `langchain-huggingface` |
| Vector store | LangChain `FAISS` — persisted as `index.faiss` + `index.pkl` |
| Retrieval | `similarity_search_with_score` with metadata source filtering |
| Prompt | LangChain `ChatPromptTemplate` |
| Chain | LCEL: `prompt | llm | StrOutputParser()` |
| LLM | Google Gemini 2.5 Flash via `langchain-google-genai` |
| API | FastAPI + uvicorn |

---

## Architecture & Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                        build_index.py                            │
│                                                                  │
│  policy.pdf ──► PyPDFLoader ──┐                                  │
│  sales.csv  ──► pd.read_csv  ─┼──► RecursiveCharacterText    ──► │
│  web URL    ──► requests/BS4 ─┘     Splitter (500/50)            │
│                                          │                       │
│                               HuggingFaceEmbeddings              │
│                               (all-MiniLM-L6-v2)                 │
│                                          │                       │
│                               FAISS.from_documents()             │
│                               .save_local("index/")              │
└──────────────────────────────────────────────────────────────────┘
                                    │
                         index.faiss + index.pkl
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                        rag_engine.py                             │
│                                                                  │
│  Query ──► similarity_search_with_score()                        │
│            + optional metadata filter {"source": ...}            │
│                          │                                       │
│            ChatPromptTemplate                                    │
│                   | (LCEL pipe)                                  │
│            ChatGoogleGenerativeAI (gemini-2.5-flash)             │
│                   | (LCEL pipe)                                  │
│            StrOutputParser()                                     │
└──────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                           api.py                                 │
│                     FastAPI REST layer                           │
│   GET /health    GET /sources    POST /query                     │
└──────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
Create a .env file and add your Google API Key
```bash
GEMINI_API_KEY=your_key_here
```

### 3. Build the index
Place your files in /data and run:
```bash
python build_index.py
```
Creates an `index/` folder containing `index.faiss` and `index.pkl`.

### 4. Run the CLI
```bash
python rag_engine.py
```

### 5. (Optional) Run the REST API
```bash
uvicorn api:app --reload
```