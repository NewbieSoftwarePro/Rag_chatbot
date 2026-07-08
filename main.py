import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for now, allow any origin — fine for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once when the server starts, not on every request
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="rice_papers",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    results = vectorstore.similarity_search(request.question, k=3)
    context = "\n\n".join(doc.page_content for doc in results)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know.

Context:
{context}
Question: {request.question}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}