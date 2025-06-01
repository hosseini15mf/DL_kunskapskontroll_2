import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
from pypdf import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import subprocess
import time

# response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": "Hello, Llama!"}])

local_model = "llama3.2"
api_model = "llama-3.3-70b-versatile"
api_url = "https://api.groq.com/openai/v1/chat/completions"
api_key = ""


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=200, chunk_overlap=40):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print("Chunking completed.")
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    print("Embedding completed.")
    return embeddings

def semantic_search(query, chunks, chunk_embeddings, model_name="all-MiniLM-L6-v2", top_k=5):
    print("Semantic searching ...")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    print("Semantic search completed.")
    return "\n".join(top_chunks)

def pull_local_model(model=local_model):
    try:
        ollama.pull(model)
        print(f"{model} model pulled successfully.")
    except Exception as e:
        print(f"Error pulling {model}:", e)

def call_local_model(prompt, model=local_model):
    pull_local_model(model)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']

def call_api_model(prompt, api_url, api_key=None, api_model=api_model):
    """
    Handles calling the remote API model.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        api_url,
        headers=headers,
        json={
            "model": api_model,
            "prompt": prompt
        },
        timeout=10
    )
    response.raise_for_status()
    api_result = response.json()

    # Support OpenAI-style or custom response formats
    if "response" in api_result:
        return api_result["response"]
    elif "choices" in api_result and isinstance(api_result["choices"], list):
        return api_result["choices"][0].get("message", {}).get("content", "No content")
    else:
        raise ValueError("Unrecognized API response format.")

def evaluate_relevance(response, reference_answer, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    emb1 = model.encode([response])[0]
    emb2 = model.encode([reference_answer])[0]
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return similarity  # closer to 1.0 is more relevant

def detect_hallucination(response, context):
    return all(sentence.lower() in context.lower() for sentence in response.split('.') if sentence.strip())

def measure_latency(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def collect_user_feedback():
    score = input("Rate response from 1 (bad) to 5 (great): ")
    return int(score)

def answer_length(response):
    return len(response.split())

def coverage_score(response, context):
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    return len(response_words & context_words) / len(response_words)

def get_response(prompt, api_url=api_url, api_key=None, api_model=api_model, local_model=local_model):
    """
    Get a response from LLaMA. Prefer API, fallback to local Ollama model if API fails.

    Returns:
        str: The model's response.
    """
    # Try API first
    if api_url:
        try:
            return call_api_model(prompt, api_url, api_key, api_model)
        except Exception as e:
            print(f"[API Error] {e} – falling back to local model.")

    # Fallback: local Ollama model
    try:
        return call_local_model(prompt, local_model)
    except Exception as e:
        print(f"[Local Model Error] {e}")
        return "Error: Could not get a response from either the API or the local model."
