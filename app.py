import streamlit as st
from openai import OpenAI

st.title("🧮 Mathe-KI-Agent")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

frage = st.text_input("Gib hier deine Mathefrage ein:")

if frage:
    with st.spinner("GPT denkt nach..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": frage}]
        )
        antwort = response.choices[0].message.content
        st.markdown("### Antwort:")
        st.write(antwort)
