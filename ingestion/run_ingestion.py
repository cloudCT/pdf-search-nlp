# Entry point for the ingestion container: fetches metadata, downloads PDFs, processes text, embeds, and uploads to Qdrant/Elasticsearch
# This script should orchestrate the full pipeline. Adjust logic as needed for your project.

import os
import sys
import logging

# Add project root for src imports if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


### Main imports

# Vector and search engine clients:
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

# Embedding model imports:
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Internal imports:
from src.utils.config import DATA_PATH

from src.data.filter_metadata import filter_metadata
from src.data.fetch_arxiv import fetch_metadata, save_metadata_to_csv, download_pdfs
from src.data.process_pdfs import process_pdfs

from src.vector_db.qdrant import create_collection
from src.sparse_db.es_client import create_index, ingest_metadata



# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Qdrant client and model
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
client = QdrantClient(qdrant_host, port=6333)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Define constants
collection_name = "corpus_embeddings"
vector_size = 384  # MiniLM output size
METADATA_CSV = os.path.join(DATA_PATH, "metadata.csv")
INDEX_NAME = "metadata_sparse"
ES_HOST = "http://elasticsearch:9200"


# Create Qdrant collection
create_collection(collection_name, vector_size)

# Elasticsearch setup
es = Elasticsearch(ES_HOST)

# Define the index mapping
# Note: Excluded "url" field
index_mapping = {
    "mappings": {
        "properties": {
            "id":          {"type": "keyword"},
            "title":       {"type": "text"},
            "abstract":    {"type": "text"},
            "categories":  {"type": "keyword"},
            "doi":         {"type": "keyword"},
            "created":     {"type": "date"},
            "updated":     {"type": "date"},
            "authors":     {"type": "text"},
            "affiliation": {"type": "text"}
        }
    }
}




### Main function

def main():
    logging.info("Starting ingestion pipeline...")

    ## Fetch metadata
    logging.info("Fetching metadata...")
    metadata_df, _ = fetch_metadata() # Note: id_list omitted for testing
    logging.info(f"Fetched {len(metadata_df)} records.")

    ## Sample and save metadata
    # Note: - Sample of 1000 id's is used for testing
    #       - use download_pdfs with id_list from fetch_metadata for full dataset
    logging.info("Sampling and saving metadata...")
    sample_df = metadata_df.sample(n=1000, random_state=42)
    sample_ids = sample_df['id'].tolist()
    save_metadata_to_csv(sample_df)

    ## Download PDFs
    logging.info("Downloading PDFs...")
    download_pdfs(sample_ids)

    ## Filter metadata
    logging.info("Filtering metadata...")
    filter_metadata()

    ## Process PDFs
    logging.info("Processing PDFs...")
    process_pdfs()

    ## Filter metadata again
    logging.info("Filtering metadata based on processed files...")
    filter_metadata()

    ## Create Qdrant collection and upload embeddings
    logging.info("Creating Qdrant collection and uploading embeddings...")
    create_collection(collection_name, vector_size)

    ## Create Elasticsearch index and ingest metadata
    logging.info("Creating Elasticsearch index and ingesting metadata...")
    create_index(es, INDEX_NAME)
    ingest_metadata(es, INDEX_NAME, METADATA_CSV)

    logging.info("Ingestion pipeline complete.")




if __name__ == "__main__":
    main()
