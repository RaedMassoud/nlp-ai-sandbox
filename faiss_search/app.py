import streamlit as st
from similarity_engine import load_data, embeddings, build_index, search_index

# laod and prep data
st.session_state.setdefault('texts', load_data())
st.session_state.setdefault('embeddings', embeddings(st.session_state.texts))
st.session_state.setdefault('index', build_index(st.session_state.embeddings))

st.title("Text Similarity Search With FAISS")

query = st.text_input("Enter your query text:")

if query:
    results = search_index(query, st.session_state.texts, st.session_state.index, top_k=5)
    st.subheader("Top Matches:")
    for match_text, score in results:
        st.write(f"- **{match_text}** _(Distance: {score:.4f})_")