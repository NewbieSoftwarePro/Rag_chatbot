from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="rice_papers",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

query = "What are the dataset name of the paper included?"

# with_score gives you the actual distance number, not just the text
results = vectorstore.similarity_search_with_score(query, k=8)

for i, (doc, score) in enumerate(results):
    print(f"--- Result {i+1} | score: {score:.4f} | {doc.metadata.get('source')} ---")
    print(doc.page_content[:200])
    print()