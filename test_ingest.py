import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data_folder = "data"
all_docs = []

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(data_folder, filename)
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {filename}: {len(docs)} pages")

print("Total pages across all PDFs:", len(all_docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(all_docs)

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rice_papers",
    persist_directory="chroma_db",
)

print("Chunks stored in Chroma:", vectorstore._collection.count())