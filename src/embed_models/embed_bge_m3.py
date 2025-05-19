# Embedding Script: BAAI/bge-m3
# Chunks and embeds scientific papers using BGE-M3.
# Default chunk size: 384 (override with --chunk_size)


This script chunks and generates embeddings for scientific papers using the BAAI/bge-m3 model.

Features:
- Uses the pretrained tokenizer for chunking.
- Processes all .txt files in the configured processed text directory.
- Saves embeddings and chunk metadata to disk for downstream search or RAG pipelines.

Usage:
    python embed_bge_m3.py --chunk_size <int> --overlap <int>

Example:
    python embed_bge_m3.py --chunk_size 384 --overlap 64

Dependencies:
    pip install torch transformers sentence-transformers tqdm numpy pandas

See README.md for more details.
"""

import sys
import os
# Robustly add project root to sys.path for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
# Main imports
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.utils.config import DATA_PATH


from src.embed_models.embed_utils import stream_chunk_text, embed_chunks

def process_corpus_bge(chunk_size=2048, overlap=64, batch_size=2, device="mps"):
    """
    Processes all .txt files in data/processed/, chunks them, embeds them, and saves embeddings + metadata to data/embeddings/bge_m3/.
    Args:
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Overlap in tokens between chunks.
        batch_size (int): Batch size for embedding.
        device (str): Device to use ('mps', 'cpu', or 'cuda'). Default 'mps'.
    """
    input_dir = os.path.join(DATA_PATH, "processed")
    output_dir = os.path.join(DATA_PATH, "embeddings", "bge_m3")
    print("Loading model: BAAI/bge-m3")
    model = SentenceTransformer("BAAI/bge-m3")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print("Found " + str(len(files)) + " text files to process.")
    all_metadata = []
    for fname in tqdm(files, desc="Embedding files"):
        paper_id = os.path.splitext(fname)[0]
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = stream_chunk_text(text, tokenizer, chunk_size=2048, max_length=2048)
        embeddings = embed_chunks(chunks, model, batch_size=2, device=device)
        np.save(os.path.join(output_dir, f"{paper_id}_embeddings.npy"), embeddings)
        for idx, chunk in enumerate(chunks):
            all_metadata.append({"paper_id": paper_id, "chunk_id": idx, "chunk": chunk})
    pd.DataFrame(all_metadata).to_csv(os.path.join(output_dir, "embedding_metadata.csv"), index=False)
    for fname in tqdm(files, desc="Embedding files"):
        paper_id = os.path.splitext(fname)[0]
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, tokenizer, chunk_size, overlap)
        embeddings = embed_chunks(chunks, model, batch_size=batch_size, device=device)
        # Save embeddings and metadata
        np.save(os.path.join(output_dir, f"{paper_id}_embeddings.npy"), embeddings)
        for idx, chunk in enumerate(chunks):
            all_metadata.append({
                "paper_id": paper_id,
                "chunk_id": idx,
                "chunk_text": chunk,
                "embedding_file": f"{paper_id}_embeddings.npy",
            })
    # Save metadata as CSV
    pd.DataFrame(all_metadata).to_csv(os.path.join(output_dir, "embedding_metadata.csv"), index=False)
    print("Done. Embeddings and metadata saved to " + output_dir)


## Function call for testing

process_corpus_bge(384, 64)



### Main function to run as script

def main():
    parser = argparse.ArgumentParser(description="Chunk and embed scientific papers using BAAI/bge-m3.")
    parser.add_argument("--chunk_size", type=int, default=384, help="Chunk size (in tokens)")
    parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for embedding")
    parser.add_argument("--device", type=str, default="mps", help="Device to use: 'mps', 'cpu', or 'cuda'")
    args = parser.parse_args()
    process_corpus_bge(args.chunk_size, args.overlap, batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
