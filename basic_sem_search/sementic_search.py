from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

## Create embeddigs
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do i change my password",
    "How do i log into my account",
    "How can i contact support",
    "Where can i update my profile"
]

embeddings = model.encode(sentences)

## Seach func
def search(query, senteces, embeddings):
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    top_idx = scores.argsort()[::-1][0]
    return sentences[top_idx], scores[top_idx]

## test
result, score = search("I need help with my password", sentences, embeddings)
print(f"Top Match: {result} (score: {score:.2f})")
