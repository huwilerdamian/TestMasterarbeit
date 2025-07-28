import streamlit as st
from PIL import Image
from openai import OpenAI
import base64

st.set_page_config(page_title="📷💬 Mathe-Chat", layout="centered")
st.title("🧮 Mathe-Chatbot mit Text & Bild")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "chatverlauf" not in st.session_state:
    st.session_state.chatverlauf = []

# Eingabefelder
col1, col2 = st.columns([4, 1])
with col1:
    textfrage = st.chat_input("Stelle deine Mathefrage...")
with col2:
    bild = st.file_uploader("📷", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# Nachricht senden nur wenn Button gedrückt oder Text abgeschickt
if (textfrage or bild):
    user_content = []

    if textfrage:
        user_content.append({"type": "text", "text": textfrage})

    if bild:
        bytes_data = bild.getvalue()
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # Systemrolle + Chatverlauf
    messages = [{"role": "system", "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."}]
    messages.extend(st.session_state.chatverlauf)

    # Nachricht je nach Format korrekt anhängen
    if len(user_content) == 1 and user_content[0]["type"] == "text":
        messages.append({"role": "user", "content": user_content[0]["text"]})
    else:
        messages.append({"role": "user", "content": user_content})

    # GPT-Antwort holen
    with st.spinner("GPT denkt nach..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        antwort = response.choices[0].message.content

    # Verlauf speichern
    st.session_state.chatverlauf.append(messages[-1])
    st.session_state.chatverlauf.append({"role": "assistant", "content": antwort})

# Darstellung des Chatverlaufs
for msg in st.session_state.chatverlauf:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], str):
            st.markdown(msg["content"])
        elif isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "text":
                    st.markdown(part["text"])
                elif part["type"] == "image_url":
                    img_data = part["image_url"]["url"].split(",")[1]
                    st.image(base64.b64decode(img_data), use_column_width=True)
