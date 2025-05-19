#### Qdrant embedding pipeline
# Includes embedding of abstract and chunks of documents
# and collecting metadata for each chunk and abstract
# Upserts all to Qdrant

# Note: pdfs and metadata must first be processed and downloaded and filtered
# see src/preprocess/ for more info

# Device: mps for m1, cpu for cpu

# Currently using mini LM as embedding model for testing


## Adding project root to sys.path for src imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

### Main imports
from src.embed_models.embed_utils import stream_chunk_text, embed_chunks, embed_abstract
from src.vector_db.qdrant import upsert_embeddings, create_collection
from src.metadata.extract_metadata import extract_metadata
import pandas as pd
import logging

# Initialize model and tokenizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create Qdrant collection
collection_name = "corpus_embeddings"
vector_size = 384  # MiniLM output size
create_collection(collection_name, vector_size)


def validate_payload(payload):
    required_fields = ["id", "title", "abstract", "categories", "doi", "created", "updated", "authors", "affiliation", "url"]
    for field in required_fields:
        if payload.get(field) is None:
            logging.warning(f"Missing required field '{field}' in payload with id {payload.get('id')}")
            return False
    return True


def run_pipeline(metadata_csv_path, processed_dir, batch_size=256, device="mps"):
    metadata_df = pd.read_csv(metadata_csv_path)
    all_vectors = []
    all_payloads = []
    seen_ids = set()
    for _, row in metadata_df.iterrows():
        document_id = row['id']
        document_text_path = os.path.join(processed_dir, f"{document_id}.txt")
        if os.path.exists(document_text_path):
            with open(document_text_path, 'r') as file:
                document_text = file.read()
            metadata_fields = extract_metadata(row.to_dict())
            # Chunk and embed document text
            chunks = stream_chunk_text(document_text, tokenizer, chunk_size=512)
            chunk_vectors = embed_chunks(chunks, model, batch_size=batch_size, device=device)
            chunk_payloads = [
                {**metadata_fields, "id": f"{document_id}_chunk_{i}", "type": "chunk", "chunk_text": chunk}
                for i, chunk in enumerate(chunks)
            ]
            # Embed abstract
            abstract_vector = embed_abstract(metadata_fields["abstract"], model, tokenizer, device=device, batch_size=batch_size)
            abstract_payload = {**metadata_fields, "id": f"{document_id}_abstract", "type": "abstract"}
           
            # Add all vectors and payloads to batch
            for vec, payload in zip(chunk_vectors, chunk_payloads):
                if payload["id"] not in seen_ids and validate_payload(payload):
                    all_vectors.append(vec)
                    all_payloads.append(payload)
                    seen_ids.add(payload["id"])
            # Abstract
            if abstract_payload["id"] not in seen_ids and validate_payload(abstract_payload):
                all_vectors.append(abstract_vector)
                all_payloads.append(abstract_payload)
                seen_ids.add(abstract_payload["id"])
            # Batch upsert if batch_size reached
            while len(all_vectors) >= batch_size:
                batch_vecs = all_vectors[:batch_size]
                batch_payloads = all_payloads[:batch_size]
                try:
                    upsert_embeddings(batch_vecs, batch_payloads, collection_name)
                except Exception as e:
                    logging.error(f"Failed to upsert batch: {e}")
                all_vectors = all_vectors[batch_size:]
                all_payloads = all_payloads[batch_size:]
    # Upsert any remaining vectors
    if all_vectors:
        try:
            upsert_embeddings(all_vectors, all_payloads, collection_name)
        except Exception as e:
            logging.error(f"Failed to upsert final batch: {e}")






################################
##### Test run within script:

# Note: pdfs and metadata must first be processed and downloaded and filtered

# Imports on top of file

# Initializing model (for test run mini ml was chosen for now)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Creating Qdrant collection
collection_name = "corpus_embeddings"
vector_size = 384  # MiniLM output size
create_collection(collection_name, vector_size)

# Define paths to metadata CSV and processed text directory
metadata_csv_path = "data/metadata.csv"
processed_dir = "data/processed"



# Running pipeline
run_pipeline(metadata_csv_path, processed_dir)










##############
### Main function to run as script:

if __name__ == "__main__":
    # Define paths to metadata CSV and processed text directory
    metadata_csv_path = "data/metadata.csv"
    processed_dir = "data/processed"

    # Run the embedding pipeline
    run_pipeline(metadata_csv_path, processed_dir)
