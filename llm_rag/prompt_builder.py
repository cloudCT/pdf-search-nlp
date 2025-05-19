"""
Build LLM context and ensure token window fit.

- Assemble context from top-ranked chunks
- Check and truncate to LLM's token window
"""


# Assemble context, ensuring token window fit
def build_llm_context(chunks,tokenizer, query, max_tokens=8192):
    """
    Returns (context string, used_chunks)
    """
    context_chunks = []
    total_tokens = len(tokenizer.encode(query))  # Start with query tokens
    for chunk in chunks:
        txt = chunk['chunk_text']
        tokens = len(tokenizer.encode(txt))
        if total_tokens + tokens > max_tokens:
            break
        context_chunks.append(chunk)
        total_tokens += tokens
    context = "\n\n".join([c['chunk_text'] for c in context_chunks])
    return context, context_chunks
