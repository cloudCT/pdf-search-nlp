"""
Retrieve and rank RAG chunks for LLM context.

- Fetch chunks from Qdrant for top-k document IDs
- Rank chunks by similarity to query
"""

# Import your Qdrant client and embedding utilities
from src.vector_db.qdrant import client as qdrant_client
from src.embed_models.embed_utils import embed_query


def get_top_chunks_for_docs(
    query, 
    doc_ids, 
    collection_name,
    embed_model, 
    device="mps", 
    top_k_chunks=20):
    """
    Retrieve and rank top-k chunks for specified document IDs based on similarity to the query.
    Returns the chunk text and score.
    """
    # Embed the query
    query_vector = embed_query(query, embed_model, device=device)

    # Qdrant filter for chunks belonging to the specified doc IDs
    chunk_filter = {
        "must": [{"key": "id", "match": {"any": doc_ids}}]
    }

    # Perform similarity search on chunks with the filter
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k_chunks,
        query_filter=chunk_filter
    )

    # Extract and return the top-k chunks with their text
    top_chunks = [
        {"chunk_text": hit.payload["chunk_text"], "score": hit.score}
        for hit in results
    ]

    return top_chunks
