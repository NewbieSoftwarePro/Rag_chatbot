import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

question = "What accuracy did the MobileNetV2 model achieve?"

# Step 1: retrieve
results = vectorstore.similarity_search(question, k=3)
context = "\n\n".join(doc.page_content for doc in results)

# Step 2: build the prompt with context + question
prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
"""

# Step 3: send to LLM
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)