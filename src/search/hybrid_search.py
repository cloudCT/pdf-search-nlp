"""
Hybrid Search for PDF Corpus: Combines ElasticSearch (sparse/BM25) and dense embeddings.
Fits into the same pipeline as dense_search and two_stage_retrieval.
"""

# Add project root to sys.path for src imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


import logging
from elasticsearch import Elasticsearch
from src.embed_models.embed_utils import embed_query
from src.vector_db.qdrant import client as qdrant_client
from sentence_transformers import SentenceTransformer

# Connect to Elasticsearch
es = Elasticsearch("http://elasticsearch:9200")

# Initialize model and tokenizer (MiniLM for testing, as in pipeline)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

dense_model = SentenceTransformer(MODEL_NAME)



### Sparse search with elasticsearch
# Intial search on raw document metadata

def sparse_search_es(query, es_index = "metadata_sparse", top_k_sparse=20):
    """
    Performs sparse retrieval in Elasticsearch and returns top_k doc ids (as list of strings).
    """
    bm25_query = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "title", 
                    "abstract", 
                    "authors", 
                    "affiliation", 
                    "categories"
                ],
                "type": "best_fields"
            }
        }
    }
    bm25_results = es.search(
        index=es_index, body=bm25_query, size=top_k_sparse)["hits"]["hits"]
    doc_ids = [hit["_id"] for hit in bm25_results]
    if not doc_ids:
        logging.info("No sparse results found.")
    return doc_ids





def sparse_filter_search_es(query, es_index = "metadata_sparse", top_k_sparse=20, filters=None):
    """
    Sparse search with Elasticsearch and optional filtering.
    
    Parameters:
    query (str): Search query.
    es_index (str): Elasticsearch index.
    top_k_sparse (int): Number of top sparse results to return.
    filters (dict): Dictionary of filters. Possible keys:
        - created_exact (str): Exact creation date filter.
        - created_after (str): Creation date after filter.
        - created_before (str): Creation date before filter.
        - created_between (list): Creation date between filter.
        - updated_exact (str): Exact update date filter.
        - updated_after (str): Update date after filter.
        - updated_before (str): Update date before filter.
        - updated_between (list): Update date between filter.
        - authors_exact (str): Exact authors filter.
    
    Returns:
    list: List of doc ids.
    """
    main_query = [
        {
            "multi_match": {
                "query": query, 
                "fields": [
                    "title", 
                    "abstract", 
                    "authors",
                    "affiliation", 
                    "categories"
                ], 
                "type": "best_fields"
            }
        }
    ]

    if filters is None:
        filters = {}

    filter_list = []
    # Filters for created date field
    if filters.get("created_exact"):
        filter_list.append({"term": {"created": filters["created_exact"]}})
    if filters.get("created_after"):
        filter_list.append({"range": {"created": {"gt": filters["created_after"]}}})
    if filters.get("created_before"):
        filter_list.append({"range": {"created": {"lt": filters["created_before"]}}})
    if filters.get("created_between"):
        filter_list.append({"range": {"created": {"gte": filters["created_between"][0], "lte": filters["created_between"][1]}}})
    # Filters for updated date field
    if filters.get("updated_exact"):
        filter_list.append({"term": {"updated": filters["updated_exact"]}})
    if filters.get("updated_after"):
        filter_list.append({"range": {"updated": {"gt": filters["updated_after"]}}})
    if filters.get("updated_before"):
        filter_list.append({"range": {"updated": {"lt": filters["updated_before"]}}})
    if filters.get("updated_between"):
        filter_list.append({"range": {"updated": {"gte": filters["updated_between"][0], "lte": filters["updated_between"][1]}}})
    # Filter for exact author match
    if filters.get("authors_exact"):
        filter_list.append({"term": {"authors": filters["authors_exact"]}})

    bm25_query = {
        "query": {
            "bool": {
                "must": main_query,
                "filter": filter_list
            }
        }
    }

    bm25_results = es.search(index=es_index, body=bm25_query, size=top_k_sparse)
    doc_ids = [hit["_id"] for hit in bm25_results["hits"]["hits"]]
    return doc_ids




### Two stage dense retrieval with Qdrant
# Takes search results from first stage sparse search as input
# -> Then performs dense retrieval on the abstract embeddings, before further
# refining results with a full dense search on everything, including 
# document text chunk embeddings

def hybrid_dense_retrieval(
    query, qdrant_collection="corpus_embeddings", doc_ids, top_k_abstracts=10, top_k_dense=5, device="mps"):
    """
    Given a query and a list of doc ids, performs two-stage dense retrieval in Qdrant:
    1. Search only abstracts for those ids, get top_k_abstracts best matches.
    2. For those ids, search all chunks and return top_k_dense final dense results.
    Returns: final dense results with metadata.
    """
    if not doc_ids:
        logging.info("No doc_ids provided for dense retrieval.")
        return []
    query_vector = embed_query(query, dense_model, device=device)

    # Stage 1: Search abstracts
    abstract_filter = {
        "must": [
            {"key": "type", "match": {"value": "abstract"}},
            {"key": "id", "match": {"any": doc_ids}}
        ]
    }
    abstracts = qdrant_client.search(
        collection_name=qdrant_collection,
        query_vector=query_vector,
        limit=top_k_abstracts,
        query_filter=abstract_filter
    )
    top_abstract_ids = [hit.payload["id"] for hit in abstracts][:top_k_abstracts]
    if not top_abstract_ids:
        logging.info("No dense abstract matches found.")
        return []

    # Stage 2: Search all chunks for these doc_ids
    chunk_filter = {
        "must": [
            {"key": "id", "match": {"any": top_abstract_ids}}
        ]
    }
    chunk_results = qdrant_client.search(
        collection_name=qdrant_collection,
        query_vector=query_vector,
        limit=1000,  # Retrieve all chunks for scoring
        query_filter=chunk_filter
    )

    # Calculate scores for each document
    document_scores = {}
    for hit in chunk_results:
        doc_id = hit.payload["id"]
        score = hit.score
        if doc_id not in document_scores:
            document_scores[doc_id] = {"scores": [], "payload": hit.payload}
        document_scores[doc_id]["scores"].append(score)

    # Compute weighted scores
    weighted_scores = []
    for doc_id, data in document_scores.items():
        highest_score = max(data["scores"])
        mean_score = sum(data["scores"]) / len(data["scores"])
        weighted_score = 0.7 * highest_score + 0.3 * mean_score
        weighted_scores.append({"doc_id": doc_id, "score": weighted_score, "payload": data["payload"]})

    # Sort and return top-k documents
    top_documents = sorted(weighted_scores, key=lambda x: x["score"], reverse=True)[:top_k_dense]
    logging.info(f"Hybrid dense retrieval for '{query}' returned {len(top_documents)} documents.")
    return top_documents




########
### In script testing


## Test for hybrid retrieval without filters

es_index = "metadata_sparse"
qdrant_collection = "corpus_embeddings"
test_query = "deep learning for natural language processing"

# Sparse retrieval
doc_ids = sparse_search_es(test_query, es_index, top_k_sparse=20)

# Dense retrieval
results = hybrid_dense_retrieval(test_query, qdrant_collection, doc_ids, top_k_abstracts=10, top_k_dense=5)

# Return results (full metadata)
for i, res in enumerate(results):
    print(f"Result {i+1}: Score={res['score']:.4f}, Metadata={res['payload']}")



## Test for hybrid retrival with filters
# Note: Hardcoded for now
#     -> will later be implemented with API, so filters can be passed as arguments for each query

es_index = "metadata_sparse"
qdrant_collection = "corpus_embeddings"
test_query = "deep learning for natural language processing"



filters = {
    "created_after": "2025-04-02",
    "updated_before": "2025-04-15",
    "authors_exact": "martin buck"
}

# Sparse retrieval
doc_ids = sparse_filter_search_es(test_query, es_index, top_k_sparse=20, filters=filters)

# Dense retrieval
results = hybrid_dense_retrieval(test_query, qdrant_collection, doc_ids, top_k_abstracts=10, top_k_dense=5)

# Return results (full metadata)
for i, res in enumerate(results):
    print(f"Result {i+1}: Score={res['score']:.4f}, Metadata={res['payload']}")




#############
# Main function to run as script
if __name__ == "__main__":
    es_index = "metadata_sparse"
    qdrant_collection = "corpus_embeddings"
    query = "deep learning for natural language processing"
    # Step 1: Sparse search (top 20)
    doc_ids = sparse_search_es(query, es_index, top_k_sparse=20)
    # Step 2: Two-stage dense retrieval (top 10 abstracts, top 5 chunks)
    results = hybrid_dense_retrieval(query, qdrant_collection, doc_ids, top_k_abstracts=10, top_k_dense=5)
    for i, res in enumerate(results):
        print(f"Result {i+1}: Score={res['score']:.4f}, Title={res['payload'].get('title')}")
