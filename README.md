# RAG-Based Document Chatbot using Gemini API

An intelligent, context-aware document chatbot built with Retrieval-Augmented Generation (RAG). This system allows users to upload PDFs and interact with the content using natural language. It leverages Google's Gemini API and sentence embeddings to deliver accurate, document-grounded answers.

---

## Project Overview

This application enables semantic search over uploaded documents and generates human-like responses by combining:

- PDF Content Extraction
- Semantic Chunk Embedding using `sentence-transformers`
- Similarity Search via FAISS (vector index)
- Generative Responses using Google's `Gemini 1.5 Pro` API

> Designed to demonstrate practical RAG architecture using scalable, modern AI tooling.

---

## Key Features

- Upload and process large PDF documents  
- Extract relevant information and context chunks  
- Ask custom questions based on document content  
- Real-time semantic search + Gemini-powered generation  
- Secure environment variable handling for API keys  
- Lightweight, fast, and interactive UI with Streamlit  

---

## Tech Stack

| Tool               | Role                                |
|--------------------|--------------------------------------|
| `Streamlit`        | Web-based interactive frontend       |
| `PyPDF2`           | PDF content extraction               |
| `sentence-transformers` | Text embedding generation         |
| `FAISS`            | Efficient semantic search            |
| `Google Gemini API`| LLM-powered content generation       |
| `dotenv`           | Environment variable handling        |


