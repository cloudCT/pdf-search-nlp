"""
LangChain-based RAG pipeline with batching and response aggregation.

- Retrieve top chunks
- Build context
- Call LLM
- Handle batching and aggregate responses
"""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from llm_rag.ollama_client import call_ollama_llm
from transformers import AutoTokenizer

# Initialize components
qdrant_client = Qdrant.from_existing_index(index_name="corpus_embeddings")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class LangChainRAG:
    def __init__(self, max_tokens=8192, top_k_chunks=20):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.max_tokens = max_tokens
        self.top_k_chunks = top_k_chunks

    def get_top_chunks(self, query, doc_ids, collection_name):
        query_vector = embed_model.embed_query(query)
        chunk_filter = {"must": [{"key": "id", "match": {"any": doc_ids}}]}
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=self.top_k_chunks,
            query_filter=chunk_filter
        )
        return [{"chunk_text": hit.payload["chunk_text"], "score": hit.score} for hit in results]

    def build_context_batches(self, chunks, query):
        context_batches = []
        current_batch = []
        total_tokens = len(self.tokenizer.encode(query))
        for chunk in chunks:
            txt = chunk['chunk_text']
            tokens = len(self.tokenizer.encode(txt))
            if total_tokens + tokens > self.max_tokens:
                context_batches.append(current_batch)
                current_batch = []
                total_tokens = len(self.tokenizer.encode(query))
            current_batch.append(chunk)
            total_tokens += tokens
        if current_batch:
            context_batches.append(current_batch)
        return context_batches

    def call_llm(self, context, query):
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = call_ollama_llm(prompt)
        return response

    def aggregate_responses(self, responses):
        combined_context = "\n\n".join(responses)
        final_response = self.call_llm(combined_context, "Summarize the above information.")
        return final_response

    def rag_answer_with_aggregation(self, query, top_doc_ids, collection_name):
        all_chunks = self.get_top_chunks(query, top_doc_ids, collection_name)
        context_batches = self.build_context_batches(all_chunks, query)

        responses = []
        for batch in context_batches:
            context = "\n\n".join([c['chunk_text'] for c in batch])
            response = self.call_llm(context, query)
            responses.append(response)

        # If only one response, return it directly
        if len(responses) == 1:
            return responses[0]

        # Otherwise, aggregate responses
        return self.aggregate_responses(responses)

    def batch_process(self, queries, top_doc_ids, collection_name):
        responses = []
        for query in queries:
            response = self.rag_answer_with_aggregation(query, top_doc_ids, collection_name)
            responses.append(response)
        return responses

    def summarize_document(self, document_text):
        # Split document into manageable chunks using the tokenizer
        chunks = self.build_context_batches([{"chunk_text": document_text}], "")
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            chunk_text = "\n\n".join([c['chunk_text'] for c in chunk])
            prompt = f"Summarize the following text:\n\n{chunk_text}"
            summary = call_ollama_llm(prompt)
            summaries.append(summary)
        
        # Aggregate summaries and pass them to the LLM for a final summary
        combined_summary = " ".join(summaries)
        final_prompt = f"Provide a comprehensive summary of the following summaries:\n\n{combined_summary}"
        final_summary = call_ollama_llm(final_prompt)
        return final_summary
