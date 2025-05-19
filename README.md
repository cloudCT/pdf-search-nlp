# Searchable PDF Corpus with NLP

In this project we build a searchable corpus of academic PDFs, downloaded from arXiv. For demonstrative purposes we chose research papers from the cs.LG category(Machine Learning and AI) from the month of April 2025. In addition subsets were utilized for testing and development. 
A hybrid search pipeline is implemented, combining sparse (ElasticSearch BM25) and a two stage dense retrieval (Qdrant). 
For RAG and document summarization Mistral 7B quantized (q4_K_M) is used as LLM. This choice was made due to its lightweight nature, allowing for fast inference and local deployment. The model is pulled and served locally via Ollama.


## Project Overview

This project aims to build a robust pipeline capable of querying metadata from arXiv, downloading PDFs, extracting text, chunking and embedding text, and employing a search index. The pipeline supports Retrieval-Augmented Generation (RAG) and document summarization, all containerized using Docker for seamless deployment.

## Main Pipeline

The pipeline is designed to handle large-scale data processing with the following steps:
1. **Metadata and PDF Retrieval:** 
   - We first fetch metadata and use the document id's to download the documents as PDF's. 
   - During testing a subset of 500 documents was used. In current pipeline implementation a subset of 100 documents is used. 
   - Afterwards all metadata is downloaded and saved as csv. Then it is filtered to only include data for donwloaded PDF's.
   - Fetching was originally tried with Sickle and OAI, which, after causing issues, are now archived and gitignored.
2. **Text Extraction and Embedding:**
   - Text is cleaned, and then extracted using PyMuPDF, chunked, and embedded using various models tested for speed and performance.
3. **Storage:**
   - Metadata is stored in Elasticsearch
   - Document chunk embeddings are stored in Qdrant, which include metadata as payload
   - Note: In current pipeline we also create and store abstract embeddings
4. **Search and Retrieval:**
   - ElasticSearch is used for sparse retrieval, while Qdrant handles a two-step dense search retrieval.
   - Filters for exact search can be used. To use filters, pass a dictionary to the filters parameter in the search function or select in streamlit UI.
5. **RAG and Summarization:**
   - Mistral 7B quantized is used for RAG and document summarization, running locally.

## Backend

FastAPI was used to build the backend API, and Docker was used to containerize the application.


## Frontend

Streamlit was used as frontend implementation.

The Streamlit app provides an interactive interface for users to query the corpus, view results, and request document summaries. 

## Embedding Models

Various embedding models were tested, each configurable within the pipeline. These can be found in src/embed_models.
Embedding models considered:
- MiniLM
- MPNet
- BGE-M3
- Jina V2
- Specter2
- SciBERT

## Hardware Requirements

Running Mistral 7B locally requires a machine with sufficient resources. Ensure your setup meets these requirements for optimal performance.

The two main options for Mistral 7B to consider are:

#### Mistral 7B (q4_K_M) – 4-bit Quantization

**RAM**  
- Minimum: 8 GB  
- Recommended: 16 GB

**CPU**  
- Architecture: ARM or x86_64  
- Cores: Minimum 4, Recommended 6–8+

**GPU**  
- Optional: Apple Silicon Neural Engine or Metal backend (experimental via Ollama)

**Disk Space**  
- ~4 GB for model files

---

#### Mistral 7B (q5_K_M) – 5-bit Quantization

**RAM**  
- Minimum: 12 GB  
- Recommended: 16 GB+

**CPU**  
- Architecture: ARM or x86_64  
- Cores: Minimum 6, Recommended 8–10+

**GPU**  
- Same as above 

**Disk Space**  
- ~6 GB for model files

---

#### LLaMA 7B (q8_0) – 8-bit Quantization

> **Note:** Mistral 7B(q5) is preferred and generally performs better while being more lightweight.

**RAM**  
- Minimum: 14–16 GB  
- Recommended: 20 GB+

**CPU**  
- Architecture: x86_64 or ARM64  
- Cores: Minimum 6, Recommended 8–12

**GPU**  
- Optional: Apple Neural Engine or Metal backend (experimental support via Ollama or llama.cpp)

**Disk Space**  
- ~8–9 GB for model files



## Pipeline Execution

### Running with Docker End to End

This project is fully containerized for reproducibility and ease of deployment. Follow these steps to run the entire pipeline using Docker Compose:

#### 1. Clone the Repository

```bash
git clone https://github.com/cloudCT/pdf-search-nlp.git
https://github.com/cloudCT/pdf-search-nlp.gityourusername/pdf-search-nlp.git
cd pdf-search-nlp
```

#### 2. Set Up Environment Variables

Copy the example environment file and update it with your credentials (e.g., Hugging Face token, Elasticsearch, Qdrant, etc.):
```bash
cp .env.example .env
# Edit .env as needed
```

#### 3. Build and Start All Services

This will build and launch all containers (FastAPI backend, Streamlit frontend, Elasticsearch, Qdrant, ingestion, embedding, etc.):
```bash
docker-compose up --build
```

#### 4. Ingest Data (Metadata and PDFs)

The ingestion step is handled by the `ingestion` service. By default, ingestion will fetch metadata from arXiv and download PDFs, then filter metadata based on downloaded PDFs, process the PDFs, create embeddings for the chunked document text and abstracts, and ingest everything in Elasticsearch and Qdrant.

To manually trigger ingestion (if needed):
```bash
docker-compose run --rm ingestion python ingestion/run_ingestion.py
```

#### 5. Launch Backend and Frontend Services

- FastAPI backend: [http://localhost:8000](http://localhost:8000)
- Streamlit frontend: [http://localhost:8501](http://localhost:8501)

#### 6. Search and RAG

Once ingestion and embedding are complete, use the Streamlit UI to search the corpus and trigger RAG (Retrieval-Augmented Generation) and summarization using Mistral 7B. You can also interact with the FastAPI backend directly for programmatic access.

#### 7. Stopping the Pipeline

To stop all services:
```bash
docker-compose down
```

---

**Notes:**
- All data (PDFs, metadata, embeddings, indices) are stored in Docker volumes or mounted directories as configured in your `docker-compose.yml`.
- To re-run ingestion or embedding for a new batch of documents, simply re-run the relevant service as described above.



### Modular approach in-script

Pipeline can also be run by sequentially calling the functions or scripts. 
1. data/fetch_arxiv.py
2. data/process_pdfs.py
3. src/sparse_db/es_client.py + src/vector_db/qdrant.py
4. src/embed_models/embed_pipeline.py
5. src/search/hybrid_search.py
6. llm_rag 



## RAG Pipeline Options

The RAG pipeline can be employed with or without LangChain. LangChain was chosen for the final pipeline due to its ease of context batching and summarization. However, a modular non-LangChain option is available.

## Authentication

A Hugging Face authentication token might be required. Make sure you created a Hugging Face account and set up your general authentication token. You can find instructions on how to do this in the Hugging Face documentation.

## Testing

We used a sample of 500 documents for testing, with the Docker pipeline currently using 1000 documents from April 2025. For different document categories or data ranges, code adjustments can be made, but note that downloading takes time.

## Embedding Model Test Scripts

Test scripts are available to evaluate model speed and performance. 
Tests used: - Speed eval on 100 documents
            - MS MARCO Passage Ranking
            - BEIR Benchmark
            - STS Benchmark


## Author

Tim Conze

---

## Future Improvements

- Add a custom embed model evaluation function
        -> subset of abstracts and manual labeling for custom queries
- Add unit tests for utility and pipeline components
- Integrate conversation and feedback loops into the pipeline using LangChain and Mistral 7B

