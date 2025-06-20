# Data Wrangler + RAG Chatbot for IRS data

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot powered by AWS Bedrock and Streamlit. 

It allows users to query sections of the [IRS Internal Revenue Manual (IRM 1.1.6)](https://www.irs.gov/irm/part1/irm_01-001-006) and receive grounded, cited answers based on official IRS documentation.

---

## Features

-  End-to-end data ingestion and preprocessing from IRS.gov
-  Document chunking, enrichment, and embedding using **Hugging Face**
-  Vector database search using **ChromaDB**
-  Answer generation using **Amazon Bedrock Titan Text Premier**
-  Streamlit app with inline citations and usage metrics

---

## Libraries

- Vector Database: Chromadb
- Embedding: sentence-transformers 
- UI interface: streamlit

## Architecture

This project follows a Retrieval-Augmented Generation (RAG) architecture:

1. Data Preparation
IRS.gov documents are crawled, parsed, cleaned, and split into meaningful sections with metadata (e.g., section ID, title, date).

2. Embedding & Indexing
Each text chunk is converted to vector embeddings using Hugging Face Transformers, and stored in a local ChromaDB vector store.

3. Retrieval
When a user submits a query, its embedding is compared to document vectors to retrieve the most relevant sections.

4. Generation
The retrieved context, user query, and the instructions are sent to Amazon Titan Text Premier via Bedrock to generate a response with inline citations.

5. Interface
A Streamlit app provides a chatbot-style UI for querying, viewing answers, and seeing source citations and performance metrics.

<img width="935" alt="Screenshot 2025-06-20 at 7 03 50 PM" src="https://github.com/user-attachments/assets/2b253257-8618-4200-8443-968b0fa7eca5" />
