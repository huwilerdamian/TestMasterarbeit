import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import json
import os

# 📁 Datei für den gespeicherten Verlauf
CHAT_FILE = "chatverlauf.json"

# 🔄 Chatverlauf laden/speichern
def save_chatverlauf(verlauf):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(verlauf, f, ensure_ascii=False, indent=2)

def load_chatverlauf():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 🧠 Session initialisieren
if "chatverlauf" not in st.session_state:
    st.session_state.chatverlauf = load_chatverlauf()

# 🔁 Button zum Zurücksetzen
if st.button("🔁 Verlauf zurücksetzen"):
    st.session_state.chatverlauf = []
    if os.path.exists(CHAT_FILE):
        os.remove(CHAT_FILE)
    st.rerun()

# 🎨 Layout
st.set_page_config(page_title="📷💬 Mathe-Chat", layout="centered")
st.title("🧮 Mathe-Chatbot mit Text & Bild")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🧑‍🎓 Texteingabe
textfrage = st.chat_input("Stelle deine Mathefrage...")

# 🖼️ Optionaler Bild-Upload
bild = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"])

# 👉 Nachricht senden, wenn Text oder Bild
if textfrage or bild:
    user_message = textfrage if textfrage else ""

    # Bild vorbereiten (falls vorhanden)
    image_dict = None
    if bild:
        bytes_data = bild.getvalue()
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

    # GPT-Nachricht erstellen
    messages = [{"role": "system", "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."}]
    for m in st.session_state.chatverlauf:
        messages.append(m)

    if image_dict:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                image_dict
            ]
        })
    else:
        messages.append({"role": "user", "content": user_message})

    # 🔄 Anfrage an GPT-4o
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
    save_chatverlauf(st.session_state.chatverlauf)

# 💬 Chatverlauf anzeigen
for msg in st.session_state.chatverlauf:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
