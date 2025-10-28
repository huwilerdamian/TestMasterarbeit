import streamlit as st
from openai import OpenAI
import openai_agents

# API-Key aus Streamlit Secrets oder Umgebung
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Deine Workflow- oder Agent-ID
AGENT_ID = st.secrets["AGENT_ID"]

st.title("🧮 Mathe-Chatbot (einfacher Agent-Aufruf)")

if "session_id" not in st.session_state:
    st.session_state.session_id = openai_agents.new_session_id()

user_text = st.chat_input("Stelle deine Mathefrage...")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Agent antwortet..."):
        # ✨ Direkter Aufruf über das agents SDK
        run = openai_agents.runs.create(
            agent_id=AGENT_ID,
            input={"text": user_text},
            session=st.session_state.session_id,
        )

    with st.chat_message("assistant"):
        st.markdown(run.output_text)
