"""
RAG pipeline orchestration.

Implementation without langchain. Does not include aggregation of responses.
"""

# Import functions from your new modules
from transformers import AutoTokenizer
from llm_rag.retrieve_rag_chunks import get_top_chunks_for_docs
from llm_rag.prompt_builder import build_llm_context
from llm_rag.ollama_client import call_ollama_llm

# Orchestrate RAG pipeline
def rag_answer(
    query, 
    top_doc_ids, 
    embed_model, 
    collection_name, 
    device="mps", 
    top_k_chunks=20, 
    max_tokens=8192):

    # Initialize mistral tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    
    # Retrieve and rank chunks
    top_chunks = get_top_chunks_for_docs(query, top_doc_ids, collection_name, embed_model, device=device, top_k_chunks=top_k_chunks)
    
    # Build LLM context
    context, _ = build_llm_context(top_chunks, tokenizer, max_tokens=max_tokens)
    
    # Create prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    # Call LLM
    return call_ollama_llm(prompt)
