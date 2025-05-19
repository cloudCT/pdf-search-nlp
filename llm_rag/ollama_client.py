"""
API client for Ollama LLM (Mistral 7B).

- Send context and query to LLM
- Return generated response
"""
import requests

# Call Ollama LLM with context and query
def call_ollama_llm(prompt, model="mistral:7b-instruct-q4_K_M"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]
