import streamlit as st
from PIL import Image
import pytesseract
from openai import OpenAI

# 👉 Konfiguration
st.set_page_config(page_title="📷 Mathe-Chat mit Bild", layout="centered")
st.title("🧮 Mathe-Chatbot mit Bild-Upload & GPT-4")

# 👉 OpenAI initialisieren
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 👉 Chatverlauf speichern (Session)
if "chatverlauf" not in st.session_state:
    st.session_state.chatverlauf = []

# 👉 Bild-Upload
bild = st.file_uploader("📷 Lade ein Bild mit einer Matheaufgabe hoch", type=["png", "jpg", "jpeg"])

if bild:
    # 📸 Bild anzeigen
    image = Image.open(bild)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # 🧠 Text mit OCR extrahieren
    aufgabe = pytesseract.image_to_string(image, lang="deu")  # Falls englisch: lang="eng"

    if aufgabe.strip():
        st.markdown("### 📝 Erkannter Aufgabentext:")
        st.code(aufgabe)

        # Nutzerfrage speichern
        st.session_state.chatverlauf.append({"role": "user", "content": aufgabe})

        # GPT-4-Abfrage
        with st.spinner("GPT analysiert die Aufgabe..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Du bist ein geduldiger Mathematik-Coach für Schüler:innen der Sekundarstufe 1. Erkläre Schritt für Schritt und verständlich."}
                ] + st.session_state.chatverlauf
            )
            antwort = response.choices[0].message.content
            st.session_state.chatverlauf.append({"role": "assistant", "content": antwort})

        st.markdown("### 💬 GPT-Antwort:")
        st.markdown(antwort)

        # Verlauf anzeigen
        with st.expander("🧠 Verlauf anzeigen"):
            for msg in st.session_state.chatverlauf:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    else:
        st.warning("⚠️ Es konnte kein Text erkannt werden. Bitte ein klareres Bild hochladen.")

# 🔁 Optional: Reset-Button
if st.button("🔄 Verlauf zurücksetzen"):
    st.session_state.chatverlauf = []
    st.experimental_rerun()
