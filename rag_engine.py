import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import tiktoken

# Load API key from .env file
load_dotenv("API_KEY.env")
genai.configure(api_key=os.getenv("Gemini_API_key"))

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Count tokens using tiktoken
def get_token_count(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# Split the text into chunks
def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current = []
    count = 0

    for word in words:
        current.append(word)
        count += 1
        if count >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            count = 0

    if current:
        chunks.append(" ".join(current))

    return chunks

# Create FAISS index with embeddings
def create_faiss_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, chunks

# Search top-k similar chunks from document
def search_similar_chunks(index, embeddings, chunks, query, k=3):
    query_vector = model.encode([query])
    _, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# Ask Gemini to answer using top chunks
def ask_gemini(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer this question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    model_name = os.getenv("GEMINI_MODEL")
    gemini_model = genai.GenerativeModel(model_name)
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
