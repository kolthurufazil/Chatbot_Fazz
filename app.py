import streamlit as st
from utils import extract_text_from_pdf
from dotenv import load_dotenv
import os
import google.generativeai as genai
from rag_engine import (
    chunk_text,
    create_faiss_index,
    search_similar_chunks,
    ask_gemini,
    get_token_count
)

# Load Gemini API key from .env
load_dotenv("API_KEY.env")
genai.configure(api_key=os.getenv("Gemini_API_key"))

st.set_page_config(page_title="PDF Q&A with FAZZ", layout="centered")
st.title("Ask Questions from Your PDF and FAZZ is here to help")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
question = st.text_input("Ask a question based on the document:")

if uploaded_file:
    with st.spinner("Fazz is reading your document..."):
        doc_text = extract_text_from_pdf(uploaded_file)
        token_count = get_token_count(doc_text)
        st.info(f"This document contains about **{token_count} tokens**.")

        if token_count > 7000:
            st.warning("The document is large. Some parts may be trimmed or skipped due to token limits.")

        # Build chunks and FAISS index
        chunks = chunk_text(doc_text, max_tokens=500)
        index, vectors, stored_chunks = create_faiss_index(chunks)

    if question:
        with st.spinner("Searching document and generating answer..."):
            top_chunks = search_similar_chunks(index, vectors, stored_chunks, question)
            answer = ask_gemini(question, top_chunks)
            st.markdown("Fazz says:")
            st.write(answer)
