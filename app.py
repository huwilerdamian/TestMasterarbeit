import streamlit as st
from openai import OpenAI
from agents import Agent, Runner
from agents.session import new_session_id

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()

AGENT_ID = st.secrets["AGENT_ID"]
user_text = st.chat_input("Stelle deine Mathefrage...")

if user_text:
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.spinner("Agent antwortet..."):
        runner = Runner(client)
        result = runner.run_sync(
            agent_id=AGENT_ID,
            input={"text": user_text},
            session=st.session_state.session_id,
        )

    with st.chat_message("assistant"):
        st.markdown(result.output_text)
