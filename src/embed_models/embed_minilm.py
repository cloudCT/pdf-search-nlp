# Embedding Script: all-MiniLM-L6-v2
# Chunks and embeds scientific papers using MiniLM.
# Default chunk size: 384 (override with --chunk_size)


This script chunks and generates embeddings for scientific papers using the all-MiniLM-L6-v2 model.

Features:
- Uses the pretrained tokenizer for chunking.
- Processes all .txt files in the configured processed text directory.
- Saves embeddings and chunk metadata to disk for downstream search or RAG pipelines.

Usage:
    python embed_minilm.py --chunk_size <int> --overlap <int>

Example:
    python embed_minilm.py --chunk_size 384 --overlap 64

Dependencies:
    pip install torch transformers sentence-transformers tqdm numpy pandas

See README.md for more details.
"""


import sys
import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.utils.config import DATA_PATH
from src.embed_models.embed_utils import stream_chunk_text, embed_chunks


def process_corpus_mlm(chunk_size, overlap, batch_size=32, device="mps"):
    """
    Processes all .txt files in data/processed/, chunks them, embeds them, and saves embeddings + metadata to data/embeddings/minilm/.
    Args:
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Overlap in tokens between chunks.
        batch_size (int): Batch size for embedding.
        device (str): Device to use ('mps', 'cpu', or 'cuda'). Default 'mps'.
    """
    input_dir = os.path.join(DATA_PATH, "processed")
    output_dir = os.path.join(DATA_PATH, "embeddings", "minilm")
    print("Loading model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print("Found " + str(len(files)) + " text files to process.")
    all_metadata = []
    for fname in tqdm(files, desc="Embedding files"):
        paper_id = os.path.splitext(fname)[0]
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = stream_chunk_text(text, tokenizer, chunk_size, overlap)
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



### Model test run


# Example usage for model-specific embeddings directory:
process_corpus_mlm(384, 64, device="mps")









### Helper functions to run model:


def run_embedding_interactive(chunk_size=384, overlap=64, batch_size=16, device="mps"):
    """
    Run the embedding pipeline interactively from a terminal or notebook.
    Example usage:
        from embed_minilm import run_embedding_interactive
        run_embedding_interactive(chunk_size=384, overlap=64)
    Embeddings are always saved to data/embeddings/minilm/.
    """
    process_corpus_mlm(chunk_size, overlap, batch_size=batch_size, device=device)

def main():
    parser = argparse.ArgumentParser(description="Chunk and embed scientific papers using all-MiniLM-L6-v2.")
    parser.add_argument("--chunk_size", type=int, default=384, help="Chunk size (in tokens)")
    parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding")
    parser.add_argument("--device", type=str, default="mps", help="Device to use: 'mps', 'cpu', or 'cuda'")
    args = parser.parse_args()
    process_corpus_mlm(args.chunk_size, args.overlap, batch_size=args.batch_size, device=args.device)

if __name__ == "__main__":
    main()
