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
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI()

context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

document_texts = [
    "Tengo 24 años",
    "Vivo en Monterrey",
    "Estudio IFI",
    "El mas joto de los presentes es Cast",
    "Fran dice puro antifacto"
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

class QueryModel(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: QueryModel):
    response = generate_response(query.query)
    return {"response": response}

if __name__ == "__main__":
    if not asyncio.get_event_loop().is_running():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        loop.create_task(server.serve())
