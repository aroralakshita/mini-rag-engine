"""
api.py — FastAPI REST backend
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

# Import the logic from your new rag_engine.py
from rag_engine import answer_question, vector_store

app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class ChunkResult(BaseModel):
    text: str
    score: float

class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks: List[ChunkResult]

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "LangChain + Gemini 2.5 Flash"}

@app.get("/sources")
def get_sources():
    # Dynamically pull sources from the FAISS index metadata
    docs = vector_store.docstore._dict.values()
    unique_sources = list({d.metadata.get("source", "Unknown") for d in docs})
    return {"sources": sorted(unique_sources)}

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Calls the function we built in rag_engine.py
        result = answer_question(
            query=req.query, 
            source_filter=req.source_filter
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks=[ChunkResult(text=c["text"], score=c["score"]) for c in result["chunks"]]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)