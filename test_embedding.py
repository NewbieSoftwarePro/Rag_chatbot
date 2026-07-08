from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text1 = "bacterial leaf blight causes yield loss in paddy crops"
text2 = "what disease affects rice leaves"
text3 = "I like eating pizza on weekends"

vec1 = embeddings.embed_query(text1)
vec2 = embeddings.embed_query(text2)
vec3 = embeddings.embed_query(text3)

print("Length of one embedding vector:", len(vec1))
print("First 5 numbers of vec1:", vec1[:5])



from sklearn.metrics.pairwise import cosine_similarity

sim_1_2 = cosine_similarity([vec1], [vec2])[0][0]
sim_1_3 = cosine_similarity([vec1], [vec3])[0][0]

print("Similarity (blight sentence vs disease question):", sim_1_2)
print("Similarity (blight sentence vs pizza sentence):", sim_1_3)