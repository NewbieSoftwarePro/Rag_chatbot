from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="rice_papers",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

query = "What accuracy did the MobileNetV2 model achieve?"
results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print("Source:", doc.metadata.get("source"))
    print("Text:", doc.page_content[:300])
    print()