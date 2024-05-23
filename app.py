import streamlit as st
import requests
import faiss
import numpy as np
from transformers import (
    DPRContextEncoder, 
    DPRContextEncoderTokenizer, 
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Thread
import uvicorn

# FastAPI setup
app = FastAPI()

class QueryModel(BaseModel):
    query: str

context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

document_texts = [
    "This is the content of document 1.",
    "This is the content of document 2."
]

document_embeddings = []
for text in document_texts:
    inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    embeddings = context_encoder(**inputs).pooler_output.detach().numpy()
    document_embeddings.append(embeddings)
document_embeddings = np.vstack(document_embeddings)

index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)

question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

llm_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
llm_model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

def retrieve_documents(query, top_k=5):
    inputs = question_tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [document_texts[i] for i in indices[0]]

def generate_response(query):
    relevant_docs = retrieve_documents(query)
    context = " ".join(relevant_docs)
    inputs = llm_tokenizer.encode(query + " " + context, return_tensors='pt')
    outputs = llm_model.generate(**inputs, max_length=500)
    response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.post("/chat")
async def chat(query: QueryModel):
    response = generate_response(query.query)
    return {"response": response}

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
thread = Thread(target=run_fastapi)
thread.daemon = True
thread.start()

# Streamlit setup
st.title("RAG Chatbot with GEMMA and Streamlit")
st.write("Ask a question:")

user_input = st.text_input("Your question:")

if st.button("Get Response"):
    if user_input:
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"query": user_input},
                timeout=30  # Set a higher timeout
            )
            if response.status_code == 200:
                st.write(response.json()["response"])
            else:
                st.write(f"Error {response.status_code}: Could not get a response from the chatbot.")
        except requests.exceptions.RequestException as e:
            st.write(f"Connection error: {e}")
    else:
        st.write("Please enter a question.")
