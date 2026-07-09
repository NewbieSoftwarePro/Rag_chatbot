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


def get_all_sources():
    """Get the unique list of paper filenames stored in Chroma."""
    all_metadata = vectorstore._collection.get(include=["metadatas"])["metadatas"]
    sources = {m.get("source") for m in all_metadata if m.get("source")}
    return sorted(sources)


ALL_SOURCES = get_all_sources()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query(request: QueryRequest):
    # Retrieve a couple of the most relevant chunks from EACH paper separately,
    # instead of one global top-k search. A global search naturally clusters
    # around whichever 1-2 papers phrase things closest to the question,
    # so broad questions ("what datasets are used across these papers")
    # would silently drop papers that never made the top-k. Searching
    # per-source guarantees every paper gets a chance to contribute.
    all_results = []
    for source in ALL_SOURCES:
        matches = vectorstore.similarity_search(
            request.question,
            k=2,
            filter={"source": source},
        )
        all_results.extend(matches)

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in all_results
    )

    prompt = f"""You are answering questions about a collection of research papers.
The context below is grouped by source paper. Some passages may not be
relevant to the question — ignore those. If the question asks broadly about
"all papers" or "each paper," cover every paper that has relevant information,
not just the closest match.

If none of the passages answer the question, say you don't have enough
information — do not guess or use knowledge outside the passages.

Context:
{context}

Question: {request.question}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}