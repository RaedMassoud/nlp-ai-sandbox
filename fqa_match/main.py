import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# laod
fqa_df = pd.read_csv("fqa_data.csv")

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed fqa questions
fqa_embeddings = model.encode(fqa_df['question'].tolist())

# find best match fqa
def find_best_fqa(user_query, threshold=.5):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, fqa_embeddings)[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return {
            "matched_question": fqa_df.iloc[best_idx]['question'],
            "answer": fqa_df.iloc[best_idx]['answer'],
            "score": best_score
        }
    else:
        return {
            "matched_question": "none",
            "answer": "sorry, could not find match",
            "score": best_score
        }
    
# example usage
if __name__ == "__main__":
    user_input = input("Ask question: ")
    result = find_best_fqa(user_input)
    print(f"\nMatched Q: {result['matched_question']}")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']}")