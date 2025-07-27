import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Titel
st.title("🧮 Mathe-KI-Agent mit Gedächtnis")

# API-Key aus Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Initialisiere GPT-4-Modell über LangChain
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4",
    temperature=0.4  # optional: weniger kreativ, mehr sachlich
)

# Gedächtnis vorbereiten (für Session)
memory = ConversationBufferMemory()

# Prompt: GPT soll wie Mathelehrer antworten
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)
conversation.prompt.template = (
    "Du bist ein geduldiger Mathematik-Coach für Schüler:innen der Sekundarstufe 1. "
    "Antworte in einfachen Worten, mit Beispielen, und stelle auch Rückfragen, wenn nötig.\n\n"
    "Chatverlauf:\n{history}\nSchüler: {input}\nCoach:"
)

# Eingabefeld für SuS
frage = st.text_input("Gib hier deine Mathefrage ein:")

# Verarbeite Frage
if frage:
    with st.spinner("GPT denkt nach..."):
        antwort = conversation.run(frage)
        st.markdown("### 💬 Antwort:")
        st.write(antwort)

        # Optional: Verlauf anzeigen
        with st.expander("🧠 Verlauf (Gedächtnis anzeigen)"):
            st.write(memory.buffer)
