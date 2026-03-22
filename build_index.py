import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/"
INDEX_DIR = "index"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── 1. Scrape/Load Functions ──────────────────────────────────────────────────
def load_web_article(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return [Document(page_content=text, metadata={"source": "web_article", "type": "web"})]
    except Exception as e:
        print(f"Web fetch failed: {e}")
        return []

def load_local_data():
    docs = []
    # PDF Logic
    from langchain_community.document_loaders import PyPDFLoader
    pdf_path = os.path.join(DATA_PATH, "policy.pdf")
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    
    # CSV Logic
    csv_path = os.path.join(DATA_PATH, "sales.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        content = df.to_string(index=False)
        docs.append(Document(page_content=content, metadata={"source": "sales.csv", "type": "csv"}))
    
    return docs

# ── 2. Build Execution ────────────────────────────────────────────────────────
def build():
    print("Loading all sources...")
    all_docs = load_local_data()
    all_docs.extend(load_web_article("https://lokalise.com/blog/rag-vs-the-buzz-how-retrieval-augmented-generation-is-quietly-disrupting-ai/"))

    if not all_docs:
        print("No data found! Add policy.pdf or sales.csv to /data.")
        return

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Split into {len(chunks)} chunks.")

    # Embedding & FAISS
    print(f"Generating embeddings ({EMBED_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save in LangChain format (Creates index.faiss AND index.pkl)
    vector_store.save_local(INDEX_DIR)
    print(f"Success! Saved to '{INDEX_DIR}/' folder.")

if __name__ == "__main__":
    os.makedirs(DATA_PATH, exist_ok=True)
    build()