import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

## load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# laod data
def load_data(file_path="data.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = [line.strip() for line in f if line.strip()]
    return text

# create embedding
def embeddings(texts):
    return model.encode(texts, normalize_embeddings=True).astype("float32")

# build FAISS index
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# search top-k similar texts
def search_index(query, text, index, top_k=5):
    query_embedding = embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    return [(text[i], float(distances[0][j])) for j, i in enumerate(indices[0])]