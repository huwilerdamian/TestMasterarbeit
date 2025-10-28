import streamlit as st
from agents import Runner
from uuid import uuid4

st.set_page_config(page_title="🧮 Mathe-Chatbot", layout="centered")
st.title("🧮 Mathe-Chatbot mit Agent")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AGENT_ID = st.secrets["AGENT_ID"]

# Session-ID einmalig erzeugen
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

# Runner ohne Parameter initialisieren
runner = Runner()

user_text = st.chat_input("Stelle deine Mathefrage...")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Agent antwortet..."):
        result = runner.run_sync(
            agent_id=AGENT_ID,
            input={"text": user_text},
            session=st.session_state.session_id,
            api_key=OPENAI_API_KEY  # hier wird der Key übergeben
        )

    with st.chat_message("assistant"):
        st.markdown(result.output_text)
