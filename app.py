import streamlit as st
import requests
from langchain.chains import RAGChain
from langchain.llms import HuggingFaceLLM
from langchain.retrievers import HuggingFaceRetriever
from transformers import (
    DPRContextEncoder, 
    DPRContextEncoderTokenizer, 
    DPRQuestionEncoder, 
    DPRQuestionEncoderTokenizer, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.testclient import TestClient

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

# LangChain setup
llm = HuggingFaceLLM(model_name="google/gemma-2b")
retriever = HuggingFaceRetriever(
    question_encoder=question_encoder,
    question_tokenizer=question_tokenizer,
    context_encoder=context_encoder,
    context_tokenizer=context_tokenizer,
    documents=[{"id": str(i), "text": text} for i, text in enumerate(document_texts)]
)

rag_chain = RAGChain(llm=llm, retriever=retriever)

@app.post("/chat")
async def chat(query: QueryModel):
    response = rag_chain(query.query)
    return {"response": response}

client = TestClient(app)

# Streamlit setup
st.title("RAG Chatbot with GEMMA and Streamlit")
st.write("Ask a question:")

user_input = st.text_input("Your question:")

if st.button("Get Response"):
    if user_input:
        try:
            response = client.post("/chat", json={"query": user_input})
            if response.status_code == 200:
                st.write(response.json()["response"])
            else:
                st.write(f"Error {response.status_code}: Could not get a response from the chatbot.")
        except requests.exceptions.RequestException as e:
            st.write(f"Connection error: {e}")
    else:
        st.write("Please enter a question.")
