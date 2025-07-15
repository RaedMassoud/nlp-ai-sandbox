import os
from openai import OpenAI
import streamlit as st

# Set your key in a .env file or system environment variable
# OPENAI_API_KEY=your-api-key-here
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(page_title="Prompt Playground", layout="centered")
st.title("Prompt_gpt")

# select task
task = st.selectbox("choose as task", ["Summarization", "Q&A", "Tone Rewrite"])

# prompt tem
temp = st.slider("Temperature (Crreatity)", 0.0, 1.0, 0.7, 0.1)

# input
user_input = st.text_area("Enter your prompt")

if st.button("Generate Output"):
    if not user_input.strip():
        st.warning("Please enter prompt first")
    else:
        if task == "Summarization":
            final_prompt = f"Summarize the following: \n\n{user_input}"
        elif task == "Q&A":
            final_prompt = f"Answer the question based on this: \n\n{user_input}"
        elif task == "Tone Rewrite":
            final_prompt = f"Rewrite this in a more professional tone: \n\n{user_input}"
        else:
            final_prompt = user_input

        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature = temp
                )
                output = response.choices[0].message.content
                st.text_area("AI Response", value=output, height=300)
            except Exception as e:
                st.error(f"Error: {e}")