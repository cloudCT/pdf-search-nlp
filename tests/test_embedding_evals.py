"""
Generic and custom evaluation script for all embedding models in the project.
Evaluates on:
- MS MARCO Passage Ranking
- BEIR Benchmark
- STS Benchmark
- Model speed (100 docs)
- Boilerplate for custom retrieval eval

Run from project root for correct imports.
"""

# Add project root to sys.path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Utils imports
from src.utils.config import DATA_PATH
from src.embed_models.embed_utils import embed_chunks
from src.embed_models.embed_utils import stream_chunk_text



import time
import random
import pandas as pd
import shutil
import requests
from tabulate import tabulate

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, evaluation, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import load_dataset

# Beir imports
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval




# List of model names to evaluate (update as needed)
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "allenai/specter2_base",
    "gsarti/scibert-nli",
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v2-base-en"
]


#######
#### Model Speed Eval
# Subset of 100 documents

def eval_speed(model_name, processed_dir = os.path.join(DATA_PATH, "processed")):
    print(f"\nTiming {model_name} on 100 chunked documents from data/processed...")
    # For SciBERT, use the correct tokenizer; for others, use model_name
    if "scibert" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = SentenceTransformer(model_name)
        model.tokenizer = tokenizer
        chunk_size = 512
        batch_size = 32
    elif "mini" in model_name.lower():
        chunk_size = 512
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 32
    elif "mpnet" in model_name.lower():
        chunk_size = 384
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 32
    elif "jina" in model_name.lower():
        chunk_size = 512
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 32
    elif "bge-m3" in model_name.lower():
        chunk_size = 2048 # max of 8192 ; lower to make it easier on ram
        batch_size = 2 # because of memory issues
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "specter2" in model_name.lower():
        chunk_size = 256
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 32
    elif "jina" in model_name.lower():
        chunk_size = 512
        batch_size = 32
        model = SentenceTransformer(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    else:
        chunk_size = 384 # Default fallback
        model = SentenceTransformer(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        batch_size = 32
    files = [f for f in os.listdir(processed_dir) if f.endswith('.txt')]
    selected_files = random.sample(files, min(10, len(files)))
    total_chunks = 0
    start = time.time()
    for fname in selected_files:
        with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
            text = f.read()
            chunks = stream_chunk_text(text, tokenizer, chunk_size=chunk_size, max_length=chunk_size)
        # Sanity check: warn if any chunk is too long and print chunk statistics
        lengths = [len(tokenizer(chunk, add_special_tokens=False)["input_ids"]) for chunk in chunks]
        for i, token_len in enumerate(lengths):
            if token_len > chunk_size:
                print(f"Warning: Chunk {i} in {fname} is {token_len} tokens (max {chunk_size})")
        if lengths:
            print(f"Chunk length stats for {fname}: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
        total_chunks += len(chunks)
        _ = embed_chunks(chunks, model, batch_size=batch_size, device="mps")
    elapsed = time.time() - start
    print(f"{model_name}: {elapsed:.2f} seconds for {len(selected_files)} docs, {total_chunks} chunks")
    return {"model": model_name, "seconds": elapsed, "num_chunks": total_chunks}


speed_results = []
for model in MODEL_NAMES:
    try:
        result = eval_speed(model)
        if result is not None:
            speed_results.append(result)
    except Exception as e:
        print(f"Speed eval failed for {model}: {e}")


speed_eval_df = pd.DataFrame(speed_results)
print(tabulate(speed_eval_df, headers="keys", tablefmt="github", showindex=False))




##############
#### MS Marco Passage Ranking
# Very small subset of the full original MS MARCO dev dataset
# Using BEIR to download the data


def extract_golden_ms_marco_subset(n_queries=5):
    # Ensure beir_datasets is placed in the same directory as this script (tests/)
    beir_dir = os.path.join(project_root, "tests", "beir_datasets")
    os.makedirs(beir_dir, exist_ok=True)
    msmarco_zip_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    msmarco_dir = beir_util.download_and_unzip(msmarco_zip_url, out_dir=beir_dir)
    # Load full corpus, queries, and qrels (dev split)
    corpus, queries, qrels = GenericDataLoader(msmarco_dir).load(split="dev")
    # Find the first n_queries with at least one relevant passage
    golden_queries = []
    for qid, rels in qrels.items():
        if len(rels) > 0:
            golden_queries.append(qid)
        if len(golden_queries) == n_queries:
            break
    golden_pids = set()
    for qid in golden_queries:
        golden_pids.update(qrels[qid].keys())
    golden_corpus = {pid: corpus[pid] for pid in golden_pids}
    golden_queries_dict = {qid: queries[qid] for qid in golden_queries}
    golden_qrels = {qid: qrels[qid] for qid in golden_queries}
    return golden_corpus, golden_queries_dict, golden_qrels

# Use the golden subset for fast and robust testing
corpus, queries, qrels = extract_golden_ms_marco_subset(n_queries=100)





# MS Marco eval function

def eval_ms_marco(model_name, corpus, queries, qrels):
    print(f"\nEvaluating {model_name} on MS MARCO Passage Ranking (BEIR, dev, sample_size={len(queries)})...")
    model = SentenceTransformer(model_name)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        show_progress_bar=True
    )
    scores = evaluator(model)
    return scores


results = []
for model_name in MODEL_NAMES:
    try:
        scores = eval_ms_marco(model_name, corpus, queries, qrels)
       
        # Main metrics for tabulate table
        result_row = {"model": model_name}
        result_row.update(scores)
        results.append(result_row)
    except Exception as e:
        results.append({"model": model_name, "error": str(e)})

# Print results table
print(tabulate(results, headers="keys", tablefmt="github", showindex=False))


# Automatically delete BEIR MS MARCO dataset directory after evaluation

msmarco_dir = os.path.join(project_root, "tests","beir_datasets", "msmarco")
try:
    shutil.rmtree(msmarco_dir)
    print(f"\nDeleted BEIR MS MARCO dataset directory: {msmarco_dir}")
except Exception as e:
    print(f"\nWarning: Could not delete {msmarco_dir}: {e}")





############
#### BEIR benchmark



# List of model names to evaluate (update as needed)
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"#,
    #"allenai/specter2_base"#,
    #"gsarti/scibert-nli",
    #"BAAI/bge-m3"
]


def eval_beir_benchmark(model_names):
    beir_dir = os.path.join(project_root, "tests", "beir_datasets")
    scifact_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    beir_path = beir_util.download_and_unzip(scifact_url, out_dir=beir_dir)
    corpus, queries, qrels = GenericDataLoader(beir_path).load(split="test")
    results = []
    k_values = [1, 3, 5, 10]
    for model_name in model_names:
        try:
            print(f"\nEvaluating {model_name} on BEIR (scifact test set)...")
            from beir.retrieval import models
            from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
            beir_model = models.SentenceBERT(model_name)
            retriever = DRES(beir_model, batch_size=16)
            retrieval = EvaluateRetrieval(retriever, score_function="cos_sim", k_values=k_values)
            retrieval_results = retrieval.retrieve(corpus, queries)
            ndcg, _map, recall, precision = retrieval.evaluate(qrels, retrieval_results, k_values)
            # Store main metrics for rank 10
            row = {
                "model": model_name,
                "NDCG@10": round(ndcg["NDCG@10"], 4),
                "MAP@10": round(_map["MAP@10"], 4),
                "Recall@10": round(recall["Recall@10"], 4),
                "Precision@10": round(precision["P@10"], 4)
            }
            results.append(row)
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})
    print("\nBEIR (scifact) Results:")
    print(tabulate(results, headers="keys", tablefmt="github", showindex=False))


eval_beir_benchmark(MODEL_NAMES)





###########
#### STS benchmark

def eval_sts_benchmark_all(model_names):
    sts_dataset_path = os.path.join(project_root, "tests", "stsbenchmark.tsv.gz")
    os.makedirs(os.path.dirname(sts_dataset_path), exist_ok=True)

    if not os.path.exists(sts_dataset_path):
        url = "https://sbert.net/datasets/stsbenchmark.tsv.gz"
        print(f"Downloading STS Benchmark dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(sts_dataset_path, "wb") as f:
            f.write(response.content)
    results = []
    for model_name in model_names:
        try:
            print(f"\nEvaluating {model_name} on STS Benchmark...")
            model = SentenceTransformer(model_name)
            from sentence_transformers import InputExample, evaluation
            import gzip
            def read_sts_data(path):
                examples = []
                open_fn = gzip.open if path.endswith(".gz") else open
                with open_fn(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        splits = line.strip().split("\t")
                        # Only use the test split and correct columns
                        if splits[0].lower() != "test":
                            continue
                        try:
                            score = float(splits[5]) / 5.0
                        except Exception:
                            continue
                        examples.append(InputExample(texts=[splits[6], splits[7]], label=score))
                return examples
            sts_examples = read_sts_data(sts_dataset_path)
            evaluator = evaluation.EmbeddingSimilarityEvaluator(
                [ex.texts[0] for ex in sts_examples],
                [ex.texts[1] for ex in sts_examples],
                [ex.label for ex in sts_examples],
                name="sts"
            )
            scores = evaluator(model)
            result_row = {"model": model_name}
            result_row.update(scores)
            results.append(result_row)
        except Exception as e:
            results.append({"model": model_name, "error": str(e)})
    print("\nSTS Benchmark Results:")
    print(tabulate(results, headers="keys", tablefmt="github", showindex=False))

# Evaluate all models on STS Benchmark
eval_sts_benchmark_all(MODEL_NAMES)




#### Notes regarding final consideration

# BGE-M3, MiniLM and MPnet were considered as final candidates.



#######
#### Custom Retrieval Eval
# TODO: Implement custom retrieval eval
# Retrieval eval on pdf corpus (scientific papers on ML)


def custom_retrieval_eval(metadata_path, label_path=None):
    print("\nCustom Retrieval Evaluation")
    # TODO: Create custom labels and queries
    #      - Could create queries and labels using abstracts to facililate labeling
    pass

