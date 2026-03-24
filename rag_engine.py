"""
rag_engine.py
─────────────
Modern LangChain RAG pipeline using LCEL:
  - LangChain FAISS retriever (with metadata filtering)
  - LCEL Chain with StrOutputParser
  - Google Gemini (gemini-2.5-flash) integration
  - Interactive CLI
"""

import os
from dotenv import load_dotenv
load_dotenv()
import time

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Config ──────────────────────────────────────────────────────────────────────
INDEX_DIR   = os.getenv("INDEX_DIR", "index")
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_KEY  = os.getenv("GEMINI_API_KEY", "")
TOP_K       = 5

if not GEMINI_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found. Please check your .env file.")

# ── Load Components ───────────────────────────────────────────────────────────
print("Loading Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# allow_dangerous_deserialization is required for loading local FAISS pickels
vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.1, # Lower temperature = more precise factual answers
)

# ── Chain Construction (LCEL) ──────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a precise, helpful assistant.
Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# This chain: Takes a dict -> Formats prompt -> Calls Gemini -> Parses result to string
chain = prompt | llm | StrOutputParser()

# ── Logic ─────────────────────────────────────────────────────────────────────
def answer_question(query: str, source_filter: str | None = None) -> dict:
    """Retrieves relevant docs and generates a grounded response."""
    
    start_time = time.time()

    # 1. Retrieval
    if source_filter:
        results = vector_store.similarity_search_with_score(query, k=TOP_K, filter={"source": source_filter})
    else:
        results = vector_store.similarity_search_with_score(query, k=TOP_K)

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    # 2. Prepare Context
    context_text = "\n\n".join([f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc, _ in results])

    # 3. Execution
    answer = chain.invoke({"context": context_text, "question": query})

    latency = round(time.time() - start_time, 2)
    print(f"\nQuery Latency: {latency} seconds")

    return {
        "answer": answer,
        "sources": list(set(doc.metadata.get("source") for doc, _ in results)),
        "chunks": [{"text": doc.page_content[:100], "score": round(float(score), 4)} for doc, score in results]
    }

# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Get available sources for the UI
    all_docs = vector_store.docstore._dict.values()
    avail_srcs = sorted({d.metadata.get("source", "Unknown") for d in all_docs})

    print(f"\nRAG Engine Active | Model: Gemini 2.5 Flash")
    print(f"Index contains documents from: {avail_srcs}\n")

    while True:
        user_query = input("Question (or 'exit'): ").strip()
        if user_query.lower() in ("exit", "quit", ""): break
        
        filter_choice = input("Filter by source? (Enter to skip): ").strip()
        source = filter_choice if filter_choice in avail_srcs else None

        print("⏳ Thinking...")
        res = answer_question(user_query, source_filter=source)

        print(f"\nAnswer:\n{res['answer']}")
        print(f"\nSources: {res['sources']}\n" + "─"*30)