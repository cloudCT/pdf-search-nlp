"""
Dense retrieval search for Qdrant using MiniLM embeddings.
"""

# Add project root to sys.path for src imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from sentence_transformers import SentenceTransformer

from src.vector_db.qdrant import client
from src.embed_models.embed_utils import embed_query





# Initialize model and tokenizer (MiniLM for testing, as in pipeline)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)



# Dense search function
def dense_search(query, collection_name, top_k=5, device="mps"):
    """
    Perform dense retrieval search over Qdrant.
    Args:
        query (str): The user query string.
        collection_name (str): Qdrant collection to search.
        top_k (int): Number of results to return.
        device (str): Device for embedding (e.g., 'mps', 'cpu').
    Returns:
        List of dicts: Each dict contains payload and score.
    """
    try:
        query_vector = embed_query(query, model, device=device)
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        # Return similarity search scores and full payloads
        formatted_results = [
            {"score": hit.score, "payload": hit.payload}
            for hit in results
        ]
        logging.info(f"Dense search for '{query}' returned {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        logging.error(f"Dense search failed: {e}")
        return []




## Two stage retrieval function
# First looks at abstract embeddings, then moves on to document chunks

def two_stage_retrieval(query, collection_name, top_k_abstracts=20, top_k=5, device="mps"):
    try:
        query_vector = embed_query(query, model, device=device)
        # Stage 1: Search only abstracts
        abstract_filter = {
            "must": [
                {"key": "type", "match": {"value": "abstract"}}
            ]
        }
        abstracts = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k_abstracts,
            query_filter=abstract_filter
        )
        doc_ids = [hit.payload["id"].replace("_abstract", "") for hit in abstracts]
        # Stage 2: Search all vectors for these doc_ids
        full_filter = {
            "must": [
                {"key": "doc_id", "match": {"any": doc_ids}}
            ]
        }
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=full_filter
        )
        formatted_results = [
            {"score": hit.score, "payload": hit.payload}
            for hit in results
        ]
        logging.info(f"Two-stage retrieval for '{query}' returned {len(formatted_results)} results.")
        return formatted_results
    except Exception as e:
        logging.error(f"Two-stage retrieval failed: {e}")
        return []


######

# Test run only returning titles
collection = "corpus_embeddings"
query = "deep learning for natural language processing"
results = dense_search(query, collection)
for i, res in enumerate(results):
    print(f"Result {i+1}: Score={res['score']:.4f}, Title={res['payload'].get('title')}")


# Test run returning all metadata
collection = "corpus_embeddings"
query = "deep learning for natural language processing"
results = dense_search(query, collection)
for i, res in enumerate(results):
    print(f"Result {i+1}: Score={res['score']:.4f}, Metadata={res['payload']}")






### Main function to run as script
if __name__ == "__main__":
    # Example usage/test
    collection = "corpus_embeddings"
    query = "deep learning for natural language processing"
    results = dense_search(query, collection)
    for i, res in enumerate(results):
        print(f"Result {i+1}: Score={res['score']:.4f}, Title={res['payload'].get('title')}")
