from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), './llm_rag'))
from langchain_rag import LangChainRAG

# Add src/ to sys.path to import project modules, allowing for relative imports of project-specific code
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from search.hybrid_search import sparse_search_es, sparse_filter_search_es
from search.dense_search import hybrid_dense_retrieval

# Initialize FastAPI app
app = FastAPI(
    title="Corpus Search API",
    description="""
        Search and retrieval API for PDF corpus of arxiv research papers.

        This API provides endpoints for searching and retrieving research papers from arxiv.
        It uses a combination of sparse (ElasticSearch BM25) and dense (Qdrant) retrieval methods.
    """
)

# Pydantic models for request/response
class FilterModel(BaseModel):
    """
    Model for filter parameters.
    
    Attributes:
    created_exact (str): Exact creation date.
    created_after (str): Creation date after.
    created_before (str): Creation date before.
    created_between (List[str]): Creation date between.
    updated_exact (str): Exact update date.
    updated_after (str): Update date after.
    updated_before (str): Update date before.
    updated_between (List[str]): Update date between.
    authors_exact (str): Exact authors.
    """
    created_exact: Optional[str] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    created_between: Optional[List[str]] = None
    updated_exact: Optional[str] = None
    updated_after: Optional[str] = None
    updated_before: Optional[str] = None
    updated_between: Optional[List[str]] = None
    authors_exact: Optional[str] = None

class SparseSearchRequest(BaseModel):
    """
    Model for sparse search request.
    
    Attributes:
    query (str): Search query.
    es_index (str): Elasticsearch index (default: "metadata_sparse").
    top_k_sparse (int): Top k sparse results (default: 20, range: 1-100).
    """
    query: str
    es_index: str = Field(default="metadata_sparse")
    top_k_sparse: int = Field(default=20, ge=1, le=100)

class SparseFilterSearchRequest(SparseSearchRequest):
    """
    Model for sparse filter search request.
    
    Attributes:
    filters (FilterModel): Filter parameters.
    """
    filters: Optional[FilterModel] = None

class HybridRetrievalRequest(SparseFilterSearchRequest):
    """
    Model for hybrid retrieval request.
    
    Attributes:
    qdrant_collection (str): Qdrant collection (default: "corpus_embeddings").
    top_k_abstracts (int): Top k abstracts (default: 10, range: 1-100).
    top_k_dense (int): Top k dense results (default: 5, range: 1-100).
    """
    qdrant_collection: str = Field(default="corpus_embeddings")
    top_k_abstracts: int = Field(default=10, ge=1, le=100)
    top_k_dense: int = Field(default=5, ge=1, le=100)

class HybridRetrievalResult(BaseModel):
    """
    Model for hybrid retrieval result.
    
    Attributes:
    score (float): Result score.
    payload (dict): Result payload.
    """
    score: float
    payload: dict

class ProcessQueryRequest(BaseModel):
    """
    Model for process query request.
    
    Attributes:
    query (str): Search query.
    top_doc_ids (list): Top document IDs.
    collection_name (str): Collection name.
    use_rag (bool): Use RAG (default: True).
    """
    query: str
    top_doc_ids: list
    collection_name: str
    use_rag: bool = Field(default=True)

# CORS middleware for browser/frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
)

@app.get("/ping")
def ping():
    """
    Ping endpoint for API health check.
    
    Returns:
    dict: {"status": "ok"}
    """
    return {"status": "ok"}

@app.post("/sparse_search")
def sparse_search(req: SparseSearchRequest):
    """
    Sparse search endpoint.
    
    Args:
    req (SparseSearchRequest): Search request.
    
    Returns:
    dict: {"doc_ids": [doc_ids]}
    
    Raises:
    HTTPException: 500 if search fails.
    """
    try:
        doc_ids = sparse_search_es(req.query, req.es_index, req.top_k_sparse)
        return {"doc_ids": doc_ids}
    except Exception as e:
        logging.exception("Sparse search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sparse_filter_search")
def sparse_filter_search(req: SparseFilterSearchRequest):
    """
    Sparse filter search endpoint.
    
    Args:
    req (SparseFilterSearchRequest): Search request.
    
    Returns:
    dict: {"doc_ids": [doc_ids]}
    
    Raises:
    HTTPException: 500 if search fails.
    """
    try:
        filters_dict = req.filters.dict() if req.filters else {}
        doc_ids = sparse_filter_search_es(req.query, req.es_index, req.top_k_sparse, filters=filters_dict)
        return {"doc_ids": doc_ids}
    except Exception as e:
        logging.exception("Sparse filter search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hybrid_retrieval", response_model=List[HybridRetrievalResult])
def hybrid_retrieval(req: HybridRetrievalRequest):
    """
    Hybrid retrieval endpoint.
    
    Args:
    req (HybridRetrievalRequest): Retrieval request.
    
    Returns:
    List[HybridRetrievalResult]: List of retrieval results.
    
    Raises:
    HTTPException: 500 if retrieval fails.
    """
    try:
        filters_dict = req.filters.dict() if req.filters else {}
        doc_ids = sparse_filter_search_es(req.query, req.es_index, req.top_k_sparse, filters=filters_dict)
        results = hybrid_dense_retrieval(
            req.query,
            req.qdrant_collection,
            doc_ids,
            top_k_abstracts=req.top_k_abstracts,
            top_k_dense=req.top_k_dense
        )
        return [HybridRetrievalResult(score=r['score'], payload=r['payload']) for r in results]
    except Exception as e:
        logging.exception("Hybrid retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_query")
def process_query(req: ProcessQueryRequest):
    """
    Process query endpoint.
    
    Args:
    req (ProcessQueryRequest): Process query request.
    
    Returns:
    dict: {"response": response}
    
    Raises:
    HTTPException: 500 if processing fails.
    """
    try:
        rag_pipeline = LangChainRAG()
        if req.use_rag:
            response = rag_pipeline.rag_answer_with_aggregation(req.query, req.top_doc_ids, req.collection_name)
        else:
            # Implement non-RAG processing logic
            response = "Non-RAG processing result"
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize_document/")
async def summarize_document(document_text: str):
    try:
        rag_pipeline = LangChainRAG()
        summary = rag_pipeline.summarize_document(document_text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# root ("/") endpoint with API info
@app.get("/")
def read_root():
    """
    API root endpoint.

    Returns:
    dict: API info.

    Alternative descriptions:
    - Search and retrieval API for PDF corpus of arxiv research papers.
    - API for searching and retrieving research papers from arxiv.
    - Corpus search API for arxiv research papers.
    """
    return {"title": "Corpus Search API", "description": "Unified search API for arXiv research papers (metadata, fulltext, hybrid retrieval)."}
