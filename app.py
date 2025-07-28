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
bild = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"], key="bild_upload")

# 👉 Nachricht nur senden, wenn mindestens Text oder Bild vorhanden
if textfrage or bild:
    user_content = []
    user_message_text = ""

    # 📝 Text einbauen
    if textfrage:
        user_message_text = textfrage
        user_content.append({"type": "text", "text": textfrage})

    # 📷 Bild einbauen (wenn vorhanden)
    if bild:
        image_bytes = bild.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_dict = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        user_content.append(image_dict)

    # Chatverlauf aufbauen
    messages = [
        {
            "role": "system",
            "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."
        }
    ] + st.session_state.chatverlauf

    # Aktuelle User-Nachricht hinzufügen
    if len(user_content) == 1 and user_content[0]["type"] == "text":
        # Nur Text → als einfacher String senden
        messages.append({"role": "user", "content": user_content[0]["text"]})
    else:
        # Text + Bild oder nur Bild → als array of parts senden
        messages.append({"role": "user", "content": user_content})


    # GPT-4o API-Aufruf
    with st.spinner("GPT denkt nach..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )

    antwort = response.choices[0].message.content

    # Verlauf speichern
    st.session_state.chatverlauf.append({"role": "user", "content": user_content if len(user_content) > 1 else user_content[0]})
    st.session_state.chatverlauf.append({"role": "assistant", "content": antwort})

# 💬 Chatverlauf anzeigen
for msg in st.session_state.chatverlauf:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):  # Bei Text + Bild
            for element in msg["content"]:
                if element["type"] == "text":
                    st.markdown(element["text"])
                elif element["type"] == "image_url":
                    b64_img = element["image_url"]["url"].split(",")[1]
                    st.image(base64.b64decode(b64_img), use_column_width=True)
        else:
            st.markdown(msg["content"])
