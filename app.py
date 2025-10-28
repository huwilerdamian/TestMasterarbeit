import streamlit as st
from agents import Runner
from uuid import uuid4
import os

st.set_page_config(page_title="🧮 Mathe-Chatbot", layout="centered")
st.title("🧮 Mathe-Chatbot mit Agent + Server-Memory")

# API-Key und Agent-ID
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AGENT_ID = st.secrets["AGENT_ID"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 🚀 Session-ID einmal pro Nutzer generieren
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

runner = Runner()

# Chat-Eingabe
user_text = st.chat_input("Stelle deine Mathefrage...")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Agent antwortet..."):
        # Server-side Memory aktivieren durch Übergabe der Session-ID
        result = runner.run_sync(
            agent_id=AGENT_ID,
            input={"text": user_text},
            session_id=st.session_state.session_id,  # 🧠 hier ist die Server-Memory aktiv
        )

    with st.chat_message("assistant"):
        st.markdown(result.output_text)

# Optionaler Button, um Memory auf dem Server zurückzusetzen
if st.button("🧹 Memory löschen"):
    st.session_state.session_id = str(uuid4())
    st.success("Neue Sitzung gestartet.")
