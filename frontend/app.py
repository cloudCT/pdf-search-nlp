######## Frontend app using Streamlit


import streamlit as st
import requests
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './llm_rag'))
from langchain_rag import LangChainRAG

st.set_page_config(page_title="Corpus Search UI", layout="wide")

st.title("Corpus Search UI (arXiv)")

st.sidebar.write("Corpus: arXiv research papers (Machine Learning / AI)")
st.sidebar.write("Date range: April 2025")
st.sidebar.write("Threshold: fixed to 0.4 for relevance filtering")

API_URL = "http://localhost:8000/hybrid_retrieval"

# --- Search Form ---
with st.form("search_form"):
    # Using ', ""' to set default value for text input
    query = st.text_input("Enter your search query:", "")  # Default value is an empty string
    col1, col2 = st.columns(2)
    with col1:
        created_after = st.text_input("Created After (YYYY-MM-DD)")
        created_before = st.text_input("Created Before (YYYY-MM-DD)")
    with col2:
        updated_after = st.text_input("Updated After (YYYY-MM-DD)")
        updated_before = st.text_input("Updated Before (YYYY-MM-DD)")
    authors_exact = st.text_input("Authors (exact match)")
    top_k_dense = st.number_input("Number of search results shown", min_value=1, max_value=100, value=5)
    filter_by_threshold = st.checkbox("Filter results by score threshold")
    use_rag = st.checkbox("Use RAG", value=True)
    if filter_by_threshold:
        # Adjust the threshold value here
        threshold = 0.4
        st.write(f"Results with score >= {threshold} will be shown")
    submitted = st.form_submit_button("Search")

if submitted and query:
    filters = {}
    if created_after:
        filters["created_after"] = created_after
    if created_before:
        filters["created_before"] = created_before
    if updated_after:
        filters["updated_after"] = updated_after
    if updated_before:
        filters["updated_before"] = updated_before
    if authors_exact:
        filters["authors_exact"] = authors_exact

    top_k_sparse = top_k_dense * 4
    top_k_abstracts = top_k_dense * 2

    payload = {
        "query": query,
        "top_k_sparse": int(top_k_sparse),
        "top_k_abstracts": int(top_k_abstracts),
        "top_k_dense": int(top_k_dense),
        "filters": filters if filters else None
    }

    try:
        with st.spinner("Searching..."):
            response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            results = response.json()
            if not results:
                st.info("No results found.")
            else:
                if filter_by_threshold:
                    results = [result for result in results if result.get("score", 0) >= threshold]
                st.subheader(f"Results for: '{query}'")
                for i, result in enumerate(results, 1):
                    payload = result.get("payload", {})
                    title = payload.get("title", "[No Title]")
                    with st.expander(f"{i}. {title}"):
                        for key, value in payload.items():
                            if key != "title":
                                st.write(f"**{key.capitalize()}**: {value}")
                        st.write(f"**Score:** {result.get('score', 'N/A')}")
                        if st.button(f"Summarize {title}"):
                            document_text = payload.get("content", "")
                            rag_pipeline = LangChainRAG()
                            summary = rag_pipeline.summarize_document(document_text)
                            st.write("Summary:", summary)
                if use_rag:
                    rag_pipeline = LangChainRAG()
                    top_doc_ids = [result.get("id", "") for result in results]
                    collection_name = "corpus_embeddings"
                    response = rag_pipeline.rag_answer_with_aggregation(query, top_doc_ids, collection_name)
                    st.write("RAG Response:", response)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
