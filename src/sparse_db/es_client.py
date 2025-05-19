"""
Script to create an Elasticsearch index for sparse retrieval and upload metadata (including abstracts) from metadata.csv.
"""
import os
import sys
import logging
import pandas as pd
from elasticsearch import Elasticsearch, helpers

# Add project root for src imports if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import DATA_PATH

# Constants
METADATA_CSV = os.path.join(DATA_PATH, "metadata.csv")
INDEX_NAME = "metadata_sparse"

ES_HOST = f"http://{os.getenv('ES_HOST', 'localhost:9200')}"

# Set up logging
logging.basicConfig(level=logging.INFO)




## Define the index mapping
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



# Create index

def create_index(es, index_name):
    if es.indices.exists(index=index_name):
        logging.info(f"Index '{index_name}' already exists.")
    else:
        es.indices.create(index=index_name, body=index_mapping)
        logging.info(f"Created index '{index_name}'.")



# Ingesting metadata
def ingest_metadata(es, index_name, csv_path):
    df = pd.read_csv(csv_path)
    # Ensure the right columns are present
    required_cols = ["id", "title", "abstract", "categories", "doi", "created", "updated", "authors", "affiliation"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Prepare documents (omit 'url', and 'year')
    docs = [
        {
            "_index": index_name,
            "_id": str(row["id"]),
            "id": str(row["id"]),
            "title": row["title"],
            "authors": row["authors"],
            "abstract": row["abstract"],
            "categories": row["categories"],
            "doi": row["doi"],
            "created": row["created"],
            "updated": row["updated"],
            "affiliation": row["affiliation"]
        }
        for _, row in df.iterrows()
    ]
    # Bulk upload
    helpers.bulk(es, docs)
    logging.info(f"Uploaded {len(docs)} metadata docs to '{index_name}'.")




### Main function to run as script
if __name__ == "__main__":
    es = Elasticsearch(ES_HOST)
    create_index(es, INDEX_NAME)
    ingest_metadata(es, INDEX_NAME, METADATA_CSV)
    logging.info("Done.")
