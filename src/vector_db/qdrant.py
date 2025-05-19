from qdrant_client import QdrantClient
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)


qdrant_host = os.getenv("QDRANT_HOST", "localhost")
client = QdrantClient(qdrant_host, port=6333)


def upsert_embeddings(vectors, payloads, collection_name):
    """
    Upsert embeddings to the specified Qdrant collection.

    Args:
        vectors (list): List of embedding vectors.
        payloads (list): List of payloads containing metadata.
        collection_name (str): Name of the Qdrant collection.
    """
    points = [
        {"id": payload["id"], "vector": vector, "payload": payload}
        for vector, payload in zip(vectors, payloads)
    ]
    try:
        client.upsert(collection_name=collection_name, points=points)
        logging.info(f"Successfully upserted {len(points)} points to collection '{collection_name}'.")
    except Exception as e:
        logging.error(f"Failed to upsert points to collection '{collection_name}': {e}")


def create_collection(collection_name, vector_size):
    """
    Create or recreate a Qdrant collection with the specified vector size.

    Args:
        collection_name (str): Name of the Qdrant collection.
        vector_size (int): Size of the vectors to be stored.
    """
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance="Cosine"
        )
        logging.info(f"Collection '{collection_name}' created with vector size {vector_size}.")
    except Exception as e:
        logging.error(f"Failed to create collection '{collection_name}': {e}")
