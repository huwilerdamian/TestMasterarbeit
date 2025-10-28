import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import os
from io import BytesIO

# Streamlit page
st.set_page_config(page_title="📷💬 Mathe-Chat (Agents SDK)", layout="centered")
st.title("🧮 Mathe-Chatbot mit OpenAI Agents SDK")

# API-Key: first try Streamlit secrets, then environment variable
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY nicht gefunden. Setze st.secrets['OPENAI_API_KEY'] oder die Umgebungsvariable OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Session state: simpler in-memory history (no .json persistence)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role": "user"|"assistant", "content": ...}

# Reset button
if st.button("🔁 Verlauf zurücksetzen"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Inputs
user_text = st.chat_input("Stelle deine Mathefrage...")
uploaded_image = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"])

def build_agent_input(text: str, image_file) -> dict:
    """
    Build a payload for the Agents SDK run call. Many Agents SDKs accept a
    JSON-like input. Here we provide a text field and, if present, an inline
    base64 image string under 'image_data_url'. Adjust to your installed SDK version.
    """
    payload = {"text": text or ""}
    if image_file:
        data = image_file.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        # include the data URL so the agent (if multimodal/vision-enabled) can access it
        payload["image_data_url"] = f"data:image/jpeg;base64,{b64}"
    return payload

def render_message(msg):
    """
    Render messages stored in session_state.chat_history.
    Each message's content can be:
      - a simple string (text)
      - a dict with fields like {"text": "...", "image_data_url": "data:..."}.
    """
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    with st.chat_message(role):
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, dict):
            # text first
            text = content.get("text")
            if text:
                st.markdown(text)
            # image if present
            image_url = content.get("image_data_url") or content.get("image")
            if image_url:
                if isinstance(image_url, str) and image_url.startswith("data:"):
                    header, b64 = image_url.split(",", 1)
                    img_bytes = base64.b64decode(b64)
                    st.image(img_bytes)
                else:
                    # maybe a URL
                    st.image(image_url)
        elif isinstance(content, list):
            # list of parts
            for part in content:
                if isinstance(part, str):
                    st.markdown(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        st.markdown(text)
                    image_url = part.get("image_data_url") or part.get("image_url") or part.get("image")
                    if image_url:
                        if isinstance(image_url, str) and image_url.startswith("data:"):
                            header, b64 = image_url.split(",", 1)
                            img_bytes = base64.b64decode(b64)
                            st.image(img_bytes)
                        else:
                            st.image(image_url)
        else:
            # fallback: stringify
            st.markdown(str(content))

def run_agent_with_fallback(input_payload: dict) -> dict:
    """
    Attempt to run an OpenAI Agent via the OpenAI Agents SDK.
    Because SDK method names and payload structure can vary between versions,
    we attempt a couple of commonly used patterns, and fall back to a chat
    completion if Agents are not available in the installed SDK.

    The returned dict should contain structured content for rendering, e.g.:
      {"text": "Antwort...", "image_data_url": "data:..."} or {"text": "..."}
    """
    # Try Agents SDK pattern 1: client.agents.run(agent=..., input={...})
    try:
        # NOTE: Replace "math-coach-agent" with an actual agent id or name if you
        # have a persistent agent created. This example demonstrates a single-run call.
        # Many SDKs allow providing a system/instruction prompt directly to the run call.
        run_resp = client.agents.run(agent="math-coach-agent", input=input_payload)
        # The exact shape of run_resp depends on SDK; try to extract textual output
        if isinstance(run_resp, dict):
            # Common fields to check
            if "output" in run_resp:
                return {"text": run_resp["output"]}
            if "response" in run_resp:
                return {"text": run_resp["response"]}
            if "results" in run_resp:
                # try to join results
                if isinstance(run_resp["results"], list):
                    texts = []
                    for r in run_resp["results"]:
                        if isinstance(r, dict) and "content" in r:
                            texts.append(r["content"])
                        else:
                            texts.append(str(r))
                    return {"text": "\n\n".join(texts)}
            # as a fallback stringify the whole response
            return {"text": str(run_resp)}
    except Exception:
        # Agents API call failed or not available; fall back to chat completions
        pass

    # Fallback: use chat-style request (compatible with many OpenAI client versions)
    try:
        messages = [
            {"role": "system", "content": "Du bist ein hilfsbereiter Mathe-Coach für SuS auf Sekundarstufe 1. Erkläre klar, freundlich und mit Beispielen."},
        ]
        # Add user prompt with optional note about image
        user_text = input_payload.get("text", "")
        if "image_data_url" in input_payload:
            # inform the model that an image is available inlined as data URL
            messages.append({"role": "user", "content": f"{user_text}\n\n(Es wurde ein Bild hochgeladen. Das Bild ist als data URL beigefügt.)\n\n{input_payload.get('image_data_url')}"})
        else:
            messages.append({"role": "user", "content": user_text})

        # Try chat completions endpoint
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        text = resp.choices[0].message.content
        return {"text": text}
    except Exception as e:
        # Last resort: show error
        return {"text": f"Fehler beim Kontaktieren der API: {e}"}


# Handle sending
if user_text or uploaded_image:
    payload = build_agent_input(user_text, uploaded_image)

    # append user message to session (store structured content)
    user_entry = {"role": "user", "content": payload}
    st.session_state.chat_history.append(user_entry)

    # call agent (with fallback)
    with st.spinner("Agent denkt nach..."):
        agent_output = run_agent_with_fallback(payload)

    assistant_entry = {"role": "assistant", "content": agent_output}
    st.session_state.chat_history.append(assistant_entry)

# Render history
for msg in st.session_state.chat_history:
    render_message(msg)