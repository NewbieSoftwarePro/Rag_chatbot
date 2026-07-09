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
results = vectorstore.similarity_search(query, k=5)

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ({doc.metadata.get('source')}) ---")
    print(doc.page_content[:300])
    print()