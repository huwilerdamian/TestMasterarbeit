import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="KI-Mathe-Chat", layout="centered")
st.title("💬 Mathe-Chatbot mit GPT-4")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Chat-Verlauf initialisieren
if "chatverlauf" not in st.session_state:
    st.session_state.chatverlauf = []

# Eingabe der SuS
frage = st.chat_input("Stell deine Mathefrage...")

# Frage hinzufügen und an GPT senden
if frage:
    st.session_state.chatverlauf.append({"role": "user", "content": frage})

    with st.spinner("GPT denkt nach..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein geduldiger Mathelehrer auf Sekundarstufe 1."}
            ] + st.session_state.chatverlauf
        )
        antwort = response.choices[0].message.content
        st.session_state.chatverlauf.append({"role": "assistant", "content": antwort})

# Chatverlauf anzeigen (wie bei chatgpt.com)
for msg in st.session_state.chatverlauf:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
