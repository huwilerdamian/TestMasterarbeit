import streamlit as st
from agents import Runner
from uuid import uuid4
import os

st.set_page_config(page_title="🧮 Mathe-Chatbot", layout="centered")
st.title("🧮 Mathe-Chatbot mit echter Server-Memory")

# 🔑 API-Key und Agent-ID
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
AGENT_ID = st.secrets["AGENT_ID"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 🧠 Eine stabile Sitzungs-ID erzeugen (gleiche ID = gemeinsamer Kontext)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
st.info(f"SDK-Version: {openai_agents.__version__}")
runner = Runner()

# 💬 Texteingabe
user_text = st.chat_input("Stelle deine Mathefrage…")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Agent antwortet…"):
        # → echte SDK-Server-Memory
        result = runner.run_sync(
            agent_id=AGENT_ID,
            input={"text": user_text},
            session_id=st.session_state.session_id,
        )

    with st.chat_message("assistant"):
        st.markdown(result.output_text)

# 🔁 Memory auf dem Server zurücksetzen
if st.button("🧹 Neue Sitzung starten"):
    st.session_state.session_id = str(uuid4())
    st.success("Neue Server-Memory gestartet.")
