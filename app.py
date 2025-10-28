import streamlit as st
from PIL import Image
from openai import OpenAI
import openai
import base64
import os
from io import BytesIO
from uuid import uuid4
from typing import Any, Dict

# Streamlit page
st.set_page_config(page_title="📷💬 Mathe-Chat (Workflow/Agent via HTTP)", layout="centered")
st.title("🧮 Mathe-Chatbot mit OpenAI Workflow/Agent (HTTP fallback)")

# Show installed openai version for debugging
try:
    st.info(f"installierte openai-Version: {openai.__version__}")
except Exception:
    st.info("openai-Version nicht verfügbar")

# API-Key: first try Streamlit secrets, then environment variable
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Workflow/Agent ID (Workflow-ID from OpenAI UI)
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

# Ensure we keep only a session identifier locally to tie server-side memory
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid4())
session_id = st.session_state.agent_session_id

# Inputs
user_text = st.chat_input("Stelle deine Mathefrage...")
uploaded_image = st.file_uploader("📷 Optional: Lade ein Bild hoch", type=["png", "jpg", "jpeg"])


def build_agent_input(text: str, image_file) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"text": text or ""}
    if image_file:
        data = image_file.getvalue()
        b64 = base64.b64encode(data).decode("utf-8")
        payload["image_data_url"] = f"data:image/jpeg;base64,{b64}"
    return payload


def extract_agent_text(run_resp: Any) -> str:
    """Versucht, einen lesbaren Text aus der Workflow/Agent-Antwort zu extrahieren."""
    try:
        if isinstance(run_resp, str):
            return run_resp
        if isinstance(run_resp, dict):
            # Gängige Felder prüfen
            for key in ("output", "response", "result", "results", "content", "messages", "text"):
                if key in run_resp and run_resp[key]:
                    val = run_resp[key]
                    if isinstance(val, str):
                        return val
                    if isinstance(val, dict):
                        # nested text field
                        if "text" in val:
                            return val["text"]
                        if "content" in val:
                            return val["content"]
                    if isinstance(val, list):
                        parts = []
                        for item in val:
                            if isinstance(item, dict):
                                parts.append(item.get("content") or item.get("text") or str(item))
                            else:
                                parts.append(str(item))
                        return "\n\n".join(parts)
            # No common fields matched, try to stringify
            return str(run_resp)
        # object-like responses (SDK objects)
        if hasattr(run_resp, "output") and run_resp.output:
            return str(run_resp.output)
        if hasattr(run_resp, "response") and run_resp.response:
            return str(run_resp.response)
        if hasattr(run_resp, "to_dict"):
            return str(run_resp.to_dict())
        if hasattr(run_resp, "json"):
            try:
                j = run_resp.json()
                return extract_agent_text(j)
            except Exception:
                pass
    except Exception:
        pass
    return str(run_resp)


def run_workflow_via_http(client: OpenAI, workflow_id: str, input_payload: Dict[str, Any], session_id: str) -> Any:
    """
    Führt einen HTTP-Request aus, um einen Workflow/Agent-Run zu starten.
    Probiert typische Endpunkte durch, weil in manchen SDK-Versionen keine high-level wrappers verfügbar sind.
    Gibt das geparste JSON-Objekt oder ein SDK-Objekt zurück.
    """
    # Mögliche Pfade (werden nacheinander versucht)
    endpoints = [
        f"/v1/workflows/{workflow_id}/runs",
        f"/v1/workflows/{workflow_id}/invocations",
        f"/v1/agents/{workflow_id}/runs",
        f"/v1/workflow_runs",
        f"/v1/runs",
    ]

    last_exc = None
    for ep in endpoints:
        try:
            body = {"input": input_payload, "session": session_id}
            # client.post ist in deiner Umgebung vorhanden (laut Debug)
            resp = client.post(ep, json=body)

            # resp kann httpx.Response-ähnlich sein
            if hasattr(resp, "status_code"):
                status = resp.status_code
                try:
                    data = resp.json()
                except Exception:
                    # fallback to text
                    data = {"raw_text": resp.text}
                if 200 <= status < 300:
                    return data
                else:
                    # Server-Antwort mit Fehlercode — nützlich für Debug
                    raise RuntimeError(f"HTTP {status} vom Endpoint {ep}: {data}")
            else:
                # resp ist ein dict oder SDK-Objekt
                if isinstance(resp, dict):
                    return resp
                if hasattr(resp, "to_dict"):
                    return resp.to_dict()
                if hasattr(resp, "json"):
                    try:
                        return resp.json()
                    except Exception:
                        return str(resp)
                return str(resp)
        except Exception as e:
            last_exc = (ep, e)
            # nächster endpoint versuchen
            continue

    if last_exc:
        ep, exc = last_exc
        raise RuntimeError(f"Kein kompatibler Workflow/Agent-HTTP-Endpunkt gefunden. Letzter Versuch: {ep}, Fehler: {exc}")
    raise RuntimeError("Kein kompatibler Workflow/Agent-HTTP-Endpunkt gefunden (keine Versuche durchgeführt).")


def fetch_session_via_http(client: OpenAI, workflow_id: str, session_id: str) -> Any:
    """
    Versucht, eine Session/Memory-Information vom Server abzuholen.
    Probiert einige mögliche GET-Endpunkte durch.
    """
    endpoints = [
        f"/v1/workflows/{workflow_id}/sessions/{session_id}",
        f"/v1/workflows/{workflow_id}/runs?session={session_id}",
        f"/v1/sessions/{session_id}",
        f"/v1/runs?session={session_id}",
    ]
    last_exc = None
    for ep in endpoints:
        try:
            resp = client.get(ep)
            if hasattr(resp, "status_code"):
                status = resp.status_code
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw_text": resp.text}
                if 200 <= status < 300:
                    return data
                else:
                    raise RuntimeError(f"HTTP {status} vom Endpoint {ep}: {data}")
            else:
                if isinstance(resp, dict):
                    return resp
                if hasattr(resp, "to_dict"):
                    return resp.to_dict()
                if hasattr(resp, "json"):
                    try:
                        return resp.json()
                    except Exception:
                        return str(resp)
                return str(resp)
        except Exception as e:
            last_exc = (ep, e)
            continue

    if last_exc:
        ep, exc = last_exc
        raise RuntimeError(f"Keinen Session-Endpunkt gefunden. Letzter Versuch: {ep}, Fehler: {exc}")
    raise RuntimeError("Kein kompatibler Session-Endpunkt gefunden (keine Versuche durchgeführt).")


# When the user submits, call the Workflow/Agent via HTTP fallback and rely on server-side memory
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

    with st.spinner("Workflow/Agent wird aufgerufen und speichert Kontext in Agent-Memory..."):
        try:
            run_resp = run_workflow_via_http(client, AGENT_ID, payload, session_id)
            assistant_text = extract_agent_text(run_resp)
        except Exception as e:
            st.error(f"Agent/Workflow call fehlgeschlagen: {e}")
            assistant_text = None
            # Fallback: chat completions so the UI remains usable
            try:
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
        st.markdown(assistant_text or "(keine Antwort erhalten)")

# Button: try to load session/memory from the server for this session id
if st.button("Konversation aus Agent Memory laden"):
    try:
        mem = fetch_session_via_http(client, AGENT_ID, session_id)
        st.subheader("Agent/Workflow Memory / Session")
        st.write(mem)
    except Exception as e:
        st.error(f"Fehler beim Laden der Agent-Memory: {e}")

# Small note for operator
st.caption("Hinweis: Die App speichert nur die session_id lokal. Der eigentliche Verlauf wird (wenn möglich) serverseitig in der Workflow/Agent Memory verwaltet. Setze OPENAI_API_KEY und AGENT_ID (Workflow-ID) in st.secrets oder als Umgebungsvariablen.")