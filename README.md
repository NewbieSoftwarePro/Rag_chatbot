# 🌾 Rice Disease Research Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about rice leaf disease research papers — grounded entirely in the source documents, not general LLM knowledge.

**Live demo:** [rag-chatbot link on GitHub Pages] *(replace with your GitHub Pages URL)*
**API:** [FastAPI Cloud link] *(replace with your FastAPI Cloud URL)*

---

## What it does

Ask it a question about the ingested research papers, and it:
1. Embeds your question into a vector
2. Searches a ChromaDB vector store for the most relevant chunks of text
3. Passes those chunks to an LLM (Groq / Llama 3.3) as context
4. Returns an answer grounded in the actual papers — and says "I don't know" instead of guessing when the answer isn't in the source material

## Why RAG

A general-purpose LLM has never seen these specific papers, so it can't answer questions about their exact findings — it either declines or hallucinates. RAG solves this by retrieving the relevant passage first and forcing the model to answer from that passage.

## Architecture

```
research papers (PDF)
        │
        ▼
  PyPDFLoader → RecursiveCharacterTextSplitter (1000 char chunks, 150 overlap)
        │
        ▼
  HuggingFace hosted embeddings (384-dim vectors)
        │
        ▼
  ChromaDB (persistent vector store)
        │
        ▼
  ── query time ──
  user question → embed → retrieve top-k chunks → build prompt → Groq (Llama 3.3) → answer
        │
        ▼
  FastAPI (/query endpoint)
        │
        ▼
  HTML / CSS / JS frontend (fetch → display)
```

**Backend** and **frontend** are deployed separately: the backend is Python that needs to execute code on every request (FastAPI Cloud), while the frontend is static files that just need to be served (GitHub Pages).

## Tech stack

- **Orchestration:** LangChain (LCEL)
- **Vector store:** ChromaDB
- **Embeddings:** HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`), hosted — not run locally, to keep the deployment lightweight
- **LLM:** Groq (Llama 3.3 70B)
- **Backend:** FastAPI, deployed on FastAPI Cloud
- **Frontend:** vanilla HTML/CSS/JS, deployed on GitHub Pages
- **PDF parsing:** PyPDFLoader

## Project structure

```
├── main.py              # FastAPI app, /query endpoint
├── test_ingest.py        # loads PDFs, chunks, embeds, stores in Chroma
├── test_search.py         # standalone retrieval test
├── test_rag.py              # standalone full RAG chain test
├── test_llm.py                # standalone Groq connection test
├── test_embedding.py            # standalone embedding test
├── index.html                    # chat interface
├── requirements.txt
├── data/                           # source PDFs (not all committed — see below)
└── chroma_db/                       # persisted vector store
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Add PDFs to `data/`, then ingest:
```bash
python test_ingest.py
```

Run the API:
```bash
uvicorn main:app --reload
```

Open `index.html` in a browser (or use a Live Server extension) — make sure `API_URL` in `index.html` points to `http://127.0.0.1:8000` for local testing.

## Deployment

- Backend: [FastAPI Cloud](https://fastapicloud.com) — connects directly to this GitHub repo, auto-deploys on push to `main`
- Frontend: GitHub Pages — serving `index.html` directly from the repo root

## What I'd improve next

- Streaming responses instead of waiting for the full answer
- Source citations in the UI (which paper/page an answer came from)
- Conversation memory across multiple questions
