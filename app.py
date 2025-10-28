import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import os
from io import BytesIO
from uuid import uuid4

# Streamlit page
st.set_page_config(page_title="📷💬 Mathe-Chat (Agents SDK Memory)", layout="centered")
st.title("🧮 Mathe-Chatbot mit OpenAI Agents SDK (Agent Memory)")

# API-Key: first try Streamlit secrets, then environment variable
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Agent ID: must be provided to use Agent Memory
AGENT_ID = None
if "AGENT_ID" in st.secrets:
    AGENT_ID = st.secrets["AGENT_ID"]
else:
    AGENT_ID = os.environ.get("AGENT_ID")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY nicht gefunden. Setze st.secrets['OPENAI_API_KEY'] oder die Umgebungsvariable OPENAI_API_KEY.")
    st.stop()

if not AGENT_ID:
    st.error("AGENT_ID nicht konfiguriert. Lege einen Agenten im OpenAI-Interface an und setze AGENT_ID in st.secrets oder als Umgebungsvariable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Generate or reuse a session id so the Agent's memory is tied to this browser session.
# We store only the session identifier locally — conversation content is persisted in the Agent Memory on the OpenAI side.
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid4())

session_id = st.session_state.agent_session_id

# Inputs
user_text = st.chat_input("Stelle deine Mathefrage...")
uploaded_image = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"])

def build_agent_input(text: str, image_file) -> dict:
    payload = {"text": text or ""}
    if image_file:
        data = image_file.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        payload["image_data_url"] = f"data:image/jpeg;base64,{b64}"
    return payload

def extract_agent_text(run_resp) -> str:
    """Try to extract a textual assistant reply from a run response."""
    try:
        # many SDK versions return a dict-like response
        if isinstance(run_resp, dict):
            # check common fields
            for key in ("output", "response", "result", "results", "content"):
                if key in run_resp and run_resp[key]:
                    val = run_resp[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, dict) and "text" in val:
                        return val["text"]
                    if isinstance(val, list):
                        parts = []
                        for item in val:
                            if isinstance(item, dict) and "content" in item:
                                parts.append(item["content"])
                            else:
                                parts.append(str(item))
                        return "\n\n".join(parts)
        # try object-like response (client may return objects)
        if hasattr(run_resp, "output") and run_resp.output:
            return str(run_resp.output)
        if hasattr(run_resp, "response") and run_resp.response:
            return str(run_resp.response)
    except Exception:
        pass
    # fallback
    return str(run_resp)


# When the user submits, call the Agent and rely on Agent Memory (server-side) to persist context
if user_text or uploaded_image:
    payload = build_agent_input(user_text, uploaded_image)

    # Display the user's message locally (we do not persist the entire chat in Streamlit session)
    with st.chat_message("user"):
        if uploaded_image and not user_text:
            st.markdown("(Bild hochgeladen)")
        elif user_text:
            st.markdown(user_text)
        if uploaded_image:
            # show uploaded image immediately
            img_bytes = uploaded_image.getvalue()
            st.image(img_bytes)

    with st.spinner("Agent denkt nach und speichert den Kontext in Agent Memory..."):
        try:
            # Use the Agents SDK run call and provide a session identifier so the Agent's Memory ties exchanges together.
            # For openai-python 2.6.1 the common pattern is client.agents.run(agent=..., input=..., session=...)
            run_resp = client.agents.run(agent=AGENT_ID, input=payload, session=session_id)
            assistant_text = extract_agent_text(run_resp)
        except Exception as e:
            # If the Agents API call fails, show the error and try a chat fallback
            st.error(f"Agent call fehlgeschlagen: {e}")
            assistant_text = None
            try:
                # fallback to chat completions so the UI remains usable
                messages = [
                    {"role": "system", "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."},
                    {"role": "user", "content": payload.get("text", "") + ("\n\n(Bild als data URL beigefügt)\n" + payload.get("image_data_url", "") if payload.get("image_data_url") else "")}
                ]
                resp = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1000)
                assistant_text = resp.choices[0].message.content
            except Exception as e2:
                assistant_text = f"Fehler bei fallback Chat API: {e2}"

    # Render assistant reply (do not store full conversation locally)
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

# Optionally provide a way to view what's in Agent Memory (if supported by your Agent setup).
if st.button("Konversation aus Agent Memory laden"):
    try:
        # Attempt to fetch session/ memory for this agent session. The exact method may vary by SDK.
        # Common patterns might include client.agents.get_session(agent=AGENT_ID, session=session_id)
        # or client.sessions.get(session=session_id). Adjust based on your installed SDK.
        mem_resp = None
        # try a couple of plausible methods
        try:
            mem_resp = client.agents.get_session(agent=AGENT_ID, session=session_id)
        except Exception:
            try:
                mem_resp = client.sessions.get(id=session_id)
            except Exception:
                try:
                    mem_resp = client.agents.get(agent=AGENT_ID, session=session_id)
                except Exception:
                    mem_resp = None

        if mem_resp:
            st.subheader("Agent Memory / Session")
            st.write(mem_resp)
        else:
            st.info("Konnte Agent-Memory / Session nicht automatisch abrufen. Überprüfe deine OpenAI SDK-Version und Agent-Konfiguration.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Agent-Memory: {e}")