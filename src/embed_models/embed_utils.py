
# Add project root to sys.path for src imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np



### Main embed function to embed chunked document text
# To be used after document chunk function

def embed_chunks(chunks, model, batch_size=32, device="mps"):
    """
    Embeds a list of text chunks using the provided SentenceTransformer model.
    Args:
        chunks (list): List of text chunks.
        model (SentenceTransformer): The embedding model.
        batch_size (int): Batch size for embedding.
        device (str): Device to use ('mps', 'cpu', or 'cuda'). Default 'mps'.
    Returns:
    """
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncation=True,
            device=device
        )
        embeddings.append(batch_emb)
    return np.vstack(embeddings)


### Abstract embed function
def embed_abstract(abstract, model, tokenizer, device="mps", batch_size=32):
    # Chunk abstract if too long
    chunks = stream_chunk_text(abstract, tokenizer, chunk_size=512)
    abstract_vectors = embed_chunks(chunks, model, batch_size=batch_size, device=device)
    # Combining abstract chunks with mean pooling to get a single abstract embedding vector
    combined_vector = sum(abstract_vectors) / len(abstract_vectors)
    return combined_vector


### Query embed function

def embed_query(query, model, device="mps"):
    query_embed = model.encode(
        query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncation=True,
        device=device
    )
    return query_embed





###### Document chunk function with streaming behavior
# This solution was implemented to deal with the initial tokenization of the full document,
# and so that chunks are created based on the model's tokenizer and chunk size.
# It is not the most efficient solution, but it is the most reliable.

def stream_chunk_text(text, tokenizer, chunk_size, overlap=64, max_length=None):
    """
    Stream process the text to create overlapping chunks based on the model's tokenizer and chunk size.
    Args:
        text: Input string to chunk.
        tokenizer: HuggingFace tokenizer.
        chunk_size: Max tokens per chunk (excluding special tokens).
        overlap: Number of tokens to overlap between chunks.
        max_length: Maximum token length for each chunk.
    Returns:
        List of text chunks (str), each with no more than chunk_size tokens.
    """
    if max_length is None:
        max_length = chunk_size
    
    chunks = []
    buffer = ""
    buffer_tokens = []

    for line in text.splitlines():
        buffer += " " + line.strip()
        buffer_tokens = tokenizer(
            buffer, 
            truncation=True,  # Ensure no segment exceeds max_length
            max_length=max_length,
            add_special_tokens=False, 
            return_attention_mask=False, 
            return_tensors=None
        )["input_ids"]

        while len(buffer_tokens) >= chunk_size:
            chunk_ids = buffer_tokens[:chunk_size]
            chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk)
            buffer_tokens = buffer_tokens[chunk_size - overlap:]
            buffer = tokenizer.decode(buffer_tokens, skip_special_tokens=True)

    # Process any remaining tokens in the buffer
    if buffer_tokens:
        chunk = tokenizer.decode(buffer_tokens, skip_special_tokens=True)
        chunks.append(chunk)

    return chunks



########## Other chunk functions
### !These chunking functions are artifacts, preserved for posterity.

# Note: a simple segmentation chunk function could also be implemented instead
# of streaming. Segmentation could be done by fixed length partitioning or approximating
# model tokenization per word.

# chunk_text was tried first, issue with initial document tokenization exceeding
# max sequence length. Additionally, with truncation only the first chunk was being processed
# per document.


# def chunk_text(text, tokenizer, chunk_size, overlap=64, max_length=None):
#     """
#     Split text into overlapping chunks based on the model's tokenizer and chunk size.
#     Args:
#         text: Input string to chunk.
#         tokenizer: HuggingFace tokenizer.
#         chunk_size: Max tokens per chunk (excluding special tokens).
#         overlap: Number of tokens to overlap between chunks.
#     Returns:
#         List of text chunks (str), each with no more than chunk_size tokens.
#     """
#     # Note: type hint removed for compatibility.
#     if max_length is None:
#         max_length = chunk_size
#     input_ids = tokenizer(
#         text, 
#         truncation=True,
#         padding=True,
#         max_length=max_length,
#         add_special_tokens=False, 
#         return_attention_mask=False, 
#         return_tensors=None)["input_ids"]
#     chunks = []
#     start = 0
#     while start < len(input_ids):
#         end = min(start + chunk_size, len(input_ids))
#         chunk_ids = input_ids[start:end]
#         if not chunk_ids:
#             break
#         chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
#         # Commented out re-tokenization and trimming for testing purposes
#         # tokenized = tokenizer(
#         #     chunk,
#         #     truncation=True,
#         #     add_special_tokens=False)["input_ids"]
#         # if len(tokenized) > chunk_size:
#         #     chunk = tokenizer.decode(tokenized[:chunk_size], skip_special_tokens=True)
#         if chunk.strip():
#             chunks.append(chunk)
#         if end == len(input_ids):
#             break
#         start += chunk_size - overlap
#     return chunks

# def iterative_chunk_text_with_overlap(text, tokenizer, chunk_size, overlap=64, max_length=None):
#     """
#     Iteratively tokenize the text to create overlapping chunks based on the model's tokenizer and chunk size.
#     Args:
#         text: Input string to chunk.
#         tokenizer: HuggingFace tokenizer.
#         chunk_size: Max tokens per chunk (excluding special tokens).
#         overlap: Number of tokens to overlap between chunks.
#     Returns:
#         List of text chunks (str), each with no more than chunk_size tokens.
#     """
#     if max_length is None:
#         max_length = chunk_size
#     chunks = []
#     while text.strip():
#         # Tokenize the text with truncation to get the first chunk
#         tokenized = tokenizer(
#             text,
#             truncation=True,
#             max_length=max_length,
#             padding=True,
#             add_special_tokens=False
#         )["input_ids"]

#         # Decode the tokenized chunk
#         chunk = tokenizer.decode(tokenized, skip_special_tokens=True)
#         chunks.append(chunk)

#         # Determine the number of tokens to remove from the start of the text
#         tokens_to_remove = len(tokenized) - overlap

#         # Retokenize the chunk to find the exact text to remove
#         chunk_text = tokenizer.decode(tokenized[:tokens_to_remove], skip_special_tokens=True)

#         # Remove the processed chunk from the original text
#         text = text[len(chunk_text):].strip()

#     return chunks
