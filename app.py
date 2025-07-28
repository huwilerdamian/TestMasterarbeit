import streamlit as st
from PIL import Image
from openai import OpenAI
import base64

st.set_page_config(page_title="📷💬 Mathe-Chat", layout="centered")
st.title("🧮 Mathe-Chatbot mit Text & Bild")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "chatverlauf" not in st.session_state:
    st.session_state.chatverlauf = []

# 🧑‍🎓 Texteingabe
textfrage = st.chat_input("Stelle deine Mathefrage...")

# 🖼️ Optionaler Bild-Upload
bild = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"])

# 👉 Nachricht nur senden, wenn mindestens Text oder Bild vorhanden
if textfrage or bild:
    user_message = ""

    if textfrage:
        user_message += textfrage + "\n"

    # Bild vorbereiten, falls vorhanden
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

    # Chatverlauf aufbauen
    messages = [
        {"role": "system", "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."}
    ]

    for msg in st.session_state.chatverlauf:
        messages.append(msg)

    # Aktuelle Nachricht hinzufügen
    if image_dict:
        messages.append({"role": "user", "content": [{"type": "text", "text": user_message}, image_dict]})
    else:
        messages.append({"role": "user", "content": user_message})

    # GPT-4 Vision-Aufruf
    with st.spinner("GPT denkt nach..."):
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=1000
        )

        antwort = response.choices[0].message.content

    # Verlauf speichern
    st.session_state.chatverlauf.append({"role": "user", "content": user_message})
    st.session_state.chatverlauf.append({"role": "assistant", "content": antwort})

# 💬 Chatverlauf anzeigen
for msg in st.session_state.chatverlauf:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
