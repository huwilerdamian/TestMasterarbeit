import streamlit as st
from PIL import Image
from openai import OpenAI
import openai
import base64
import os
from io import BytesIO
from uuid import uuid4

# Streamlit page
st.set_page_config(page_title="📷💬 Mathe-Chat (Agents/Workflows SDK Memory)", layout="centered")
st.title("🧮 Mathe-Chatbot mit OpenAI Agents/Workflows SDK (Agent Memory)")

# Debug: show installed openai version
st.info(f"installierte openai-Version: {openai.__version__}")

# API-Key: first try Streamlit secrets, then environment variable
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Agent / Workflow ID: must be provided to use Agent Memory (set to your Workflow ID)
AGENT_ID = None
if "AGENT_ID" in st.secrets:
    AGENT_ID = st.secrets["AGENT_ID"]
else:
    AGENT_ID = os.environ.get("AGENT_ID")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY nicht gefunden. Setze st.secrets['OPENAI_API_KEY'] oder die Umgebungsvariable OPENAI_API_KEY.")
    st.stop()

if not AGENT_ID:
    st.error("AGENT_ID (Workflow-ID) nicht konfiguriert. Lege einen Agent/Workflow im OpenAI-Interface an und setze AGENT_ID in st.secrets oder als Umgebungsvariable.")
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
            for key in ("output", "response", "result", "results", "content", "messages"):
                if key in run_resp and run_resp[key]:
                    val = run_resp[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, dict) and "text" in val:
                        return val["text"]
                    if isinstance(val, list):
                        parts = []
                        for item in val:
                            if isinstance(item, dict) and ("content" in item or "text" in item):
                                parts.append(item.get("content") or item.get("text"))
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

def run_agent_compat(client, agent_or_workflow_id, input_payload, session_id):
    """
    Try multiple possible SDK entrypoints for Agents/Workflows.
    Returns the raw run response or raises an informative exception.
    """
    last_exc = None

    # 1) Try client.workflows.run(workflow=..., input=..., session=...)
    try:
        if hasattr(client, "workflows"):
            try:
                return client.workflows.run(workflow=agent_or_workflow_id, input=input_payload, session=session_id)
            except TypeError:
                # alternative param names
                return client.workflows.run(id=agent_or_workflow_id, input=input_payload, session=session_id)
    except Exception as e:
        last_exc = e

    # 2) Try client.agents.run(agent=..., input=..., session=...)
    try:
        if hasattr(client, "agents"):
            try:
                return client.agents.run(agent=agent_or_workflow_id, input=input_payload, session=session_id)
            except TypeError:
                return client.agents.run(id=agent_or_workflow_id, input=input_payload, session=session_id)
    except Exception as e:
        last_exc = e

    # 3) Try workflow_runs / runs style API (common alternative)
    try:
        if hasattr(client, "workflow_runs"):
            try:
                return client.workflow_runs.create(workflow=agent_or_workflow_id, input=input_payload, session=session_id)
            except TypeError:
                return client.workflow_runs.create(id=agent_or_workflow_id, input=input_payload, session=session_id)
        if hasattr(client, "runs"):
            try:
                return client.runs.create(workflow=agent_or_workflow_id, input=input_payload, session=session_id)
            except TypeError:
                return client.runs.create(id=agent_or_workflow_id, input=input_payload, session=session_id)
    except Exception as e:
        last_exc = e

    raise RuntimeError(
        "Kein kompatibler Agents-/Workflows-Endpunkt in der aktuellen openai-Python-Installation gefunden. "
        "Prüfe openai.__version__ und update ggf. mit `pip install --upgrade openai`. "
        f"Letzte Fehlermeldung: {last_exc}"
    )

# When the user submits, call the Agent/Workflow and rely on server-side Memory to persist context
if user_text or uploaded_image:
    payload = build_agent_input(user_text, uploaded_image)

    # Display the user's message locally (we do not persist the entire chat in Streamlit session)
    with st.chat_message("user"):
        if uploaded_image and not user_text:
            st.markdown("(Bild hochgeladen)")
        elif user_text:
            st.markdown(user_text)
        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            st.image(img_bytes)

    with st.spinner("Agent/Workflow denkt nach und speichert den Kontext in Agent Memory..."):
        try:
            run_resp = run_agent_compat(client, AGENT_ID, payload, session_id)
            assistant_text = extract_agent_text(run_resp)
        except Exception as e:
            st.error(f"Agent/Workflow call fehlgeschlagen: {e}")
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

# Optionally provide a way to view what's in Agent Memory (if supported by your Agent/Workflow setup).
if st.button("Konversation aus Agent Memory laden"):
    try:
        mem_resp = None
        # try a couple of plausible methods for session/memory retrieval
        try:
            if hasattr(client, "workflows") and hasattr(client.workflows, "get_session"):
                mem_resp = client.workflows.get_session(workflow=AGENT_ID, session=session_id)
        except Exception:
            pass
        try:
            if not mem_resp and hasattr(client, "agents") and hasattr(client.agents, "get_session"):
                mem_resp = client.agents.get_session(agent=AGENT_ID, session=session_id)
        except Exception:
            pass
        try:
            if not mem_resp and hasattr(client, "sessions") and hasattr(client.sessions, "get"):
                mem_resp = client.sessions.get(id=session_id)
        except Exception:
            pass

        if mem_resp:
            st.subheader("Agent/Workflow Memory / Session")
            st.write(mem_resp)
        else:
            st.info("Konnte Agent/Workflow-Memory / Session nicht automatisch abrufen. Überprüfe deine OpenAI SDK-Version und Agent/Workflow-Konfiguration.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Agent-Memory: {e}")