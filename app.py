import os
import json
import time
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import boto3

# ======================= Setup ChromaDB ============================
@st.cache_resource
def setup_chroma():
    with open("data/sections.json", "r", encoding="utf-8") as f:
        sections = json.load(f)
    texts = [s["text"] for s in sections]
    metadatas = [{"section": s["section"], "title": s["title"], "date": s["date"]} for s in sections]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)

    client = chromadb.Client()
    collection = client.create_collection(name="irs_sections")

    # Insert documents
    for i, (text, meta, embedding) in enumerate(zip(texts, metadatas, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[str(i)]
        )
    return model, collection

# ======================= Query Helpers ============================
def estimate_tokens(text): return int(len(text) / 4)


def query_chromadb(user_query, model, collection, top_k=3):
    query_embedding = model.encode(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas


def estimate_token_count(text):
    # Approx: 1 token ~ 4 characters (common for LLMs)
    return int(len(text) / 4)


def generate_answer_titan(context, user_query):
    prompt = f"""You are an assistant helping answer questions using the IRS Internal Revenue Manual.

Use the context below to answer the question.
Each document in the context below starts with a section number.
If that section is used to answer a prompt, include inline section references like [1.1.6.4].
If there are multiple sections, include all of them.
Only use the sections from the context below for referencing.
Do not use generic citations like [1], [2], etc.
Return the answer with inline citations. 

Context:
{context}

Question: {user_query}

Answer:"""

    start_time = time.time()

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 2000,
            "temperature": 0.5,
            "topP": 0.5,
            "stopSequences": []
        }
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-text-premier-v1:0",
        body=json.dumps(body),
        contentType="application/json"
    )

    latency = time.time() - start_time

    result = json.loads(response["body"].read())
    output = result["results"][0]["outputText"]

    prompt_tokens = estimate_token_count(prompt)
    output_tokens = estimate_token_count(output)

    return output, {
        "latency": f"{round(latency, 2)} (s)",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }

# ======================= Streamlit Chatbot UI ============================
st.set_page_config(page_title="IRS Chatbot", layout="centered")
st.title("💬 IRS Internal Manual Chatbot")

model, collection = setup_chroma()
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Persistent chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat message display
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something from the IRS Internal Manual..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        context, citations = query_chromadb(prompt, model, collection)
        answer, metrics = generate_answer_titan(context, prompt)

        # Format citations
        citations_text = "\n".join([f"- {c['section']} – {c['title']} ({c['date']})" for c in citations])
        metrics_text = "\n".join([f"- {m}: {metrics[m]}" for m in metrics])
        full_reply = f"{answer}\n\nSources:\n{citations_text}\n\nMetrics:\n{metrics_text}"

    st.chat_message("assistant").markdown(full_reply)
    st.session_state.chat.append({"role": "assistant", "content": full_reply})
