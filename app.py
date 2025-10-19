import os, re, json, html, base64, tempfile, io
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS

# ==========================
# 🚀 SETUP
# ==========================
load_dotenv()
st.set_page_config(page_title="Virtual Patient", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY")
    st.stop()

MODEL = "gpt-4o-mini"
CASES_DIR = Path("cases")
AVATAR_DIR = Path(".")
LOGS_DIR = Path("conversations")
LOGS_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# 🧠 HELPERS
# ==========================
def esc(x: str) -> str:
    return html.escape(x).replace("\n", "<br>")

def tts_mp3(text: str) -> str:
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def speech_to_text(audio_file) -> str:
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcription.text.strip()

def call_llm_as_patient(case: Dict, history: List[Dict[str, str]]) -> str:
    system = {
        "role": "system",
        "content": (
            "You are a human patient in a clinical interview.\n"
            "- Speak naturally and reveal information gradually.\n"
            "- If the question is vague, state your main symptom.\n"
            "- Do not act like an AI.\n\n"
            f"CASE:\n{json.dumps(case)}"
        )
    }
    msgs = [system] + [{"role": m["role"], "content": m["content"]} for m in history[-20:]]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=0.8,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# ==========================
# STATE
# ==========================
if "case" not in st.session_state: st.session_state.case = None
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None
if "history" not in st.session_state: st.session_state.history = []
if "input_mode" not in st.session_state: st.session_state.input_mode = "keyboard"
if "sent" not in st.session_state: st.session_state.sent = False

# ==========================
# 💬 CHAT PAGE
# ==========================
if st.session_state.case:
    st.button("⬅️ Back to Patients", on_click=lambda: (st.session_state.update({"case": None, "history": []}), st.rerun()))

    # Enlarged patient image with name under
    st.markdown(f"""
    <div style='display:flex;flex-direction:column;align-items:center;gap:10px;background:white;padding:20px;border-radius:15px;box-shadow:0 2px 6px rgba(0,0,0,0.05);margin-bottom:15px;'>
        <img src='data:image/png;base64,{base64.b64encode(open(st.session_state.avatar_path, "rb").read()).decode()}'
             style='border-radius:50%;width:150px;height:150px;object-fit:cover;'>
        <div style='text-align:center;'>
            <h2 style='margin:0;'>{st.session_state.patient_name}</h2>
            <div style='color:#777;font-size:14px;'>{st.session_state.case.get("title","")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Convert patient avatar to base64 for message icon
    with open(st.session_state.avatar_path, "rb") as img_f:
        patient_icon_b64 = base64.b64encode(img_f.read()).decode()

    # ==========================
    # CHAT LOOP (latest at bottom)
    # ==========================
    chat_html = "<div class='chat' style='background:#fff;border-radius:14px;height:500px;overflow-y:auto;display:flex;flex-direction:column-reverse;padding:15px;'>"
    for m in reversed(st.session_state.history):
        if m["role"] == "user":
            chat_html += f"""
            <div class='bubble doctor'>
                <span class='role'>👨‍⚕️</span>{esc(m['content'])}
            """
            if "audio" in m:
                with open(m["audio"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chat_html += f"<audio controls style='display:block;margin-top:4px;'><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
            chat_html += "</div>"
        else:
            chat_html += f"""
            <div class='bubble patient'>
                <img src='data:image/png;base64,{patient_icon_b64}'
                     style='width:25px;height:25px;border-radius:50%;vertical-align:middle;margin-right:8px;'>
                {esc(m['content'])}
            """
            if "audio" in m:
                with open(m["audio"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chat_html += f"<audio controls style='display:block;margin-top:4px;'><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
            chat_html += "</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ==========================
    # Input Bar (Mic + Keyboard)
    # ==========================
    st.markdown("""
    <style>
    .icon-button {
        background:white;
        border:1px solid #ccc;
        padding:8px 12px;
        border-radius:10px;
        cursor:pointer;
        font-size:18px;
    }
    .icon-button.active { background-color:#4B72FF; color:white; }
    </style>
    """, unsafe_allow_html=True)

    col_input, col_buttons = st.columns([0.85, 0.15])
    with col_buttons:
        kb = st.button("⌨", use_container_width=True)
        mic = st.button("🎤", use_container_width=True)
        if kb: st.session_state.input_mode = "keyboard"
        if mic: st.session_state.input_mode = "voice"

    with col_input:
        if st.session_state.input_mode == "keyboard":
            user_text = st.text_input("Type your question…", key="text_input", label_visibility="collapsed")
            if user_text and not st.session_state.sent:
                st.session_state.history.append({"role": "user", "content": user_text})
                reply = call_llm_as_patient(st.session_state.case, st.session_state.history)
                audio_path = tts_mp3(reply)
                st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                if "text_input" in st.session_state:
                    del st.session_state["text_input"]
                st.session_state.sent = True
                st.rerun()
            else:
                st.session_state.sent = False

        else:
            audio_data = st.audio_input("Record your question", label_visibility="collapsed")
            if audio_data and not st.session_state.sent:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data.read())
                    f.flush()
                    text_transcribed = speech_to_text(f.name)
                if text_transcribed:
                    st.session_state.history.append({"role": "user", "content": f"{text_transcribed}", "audio": f.name})
                    reply = call_llm_as_patient(st.session_state.case, st.session_state.history)
                    audio_path = tts_mp3(reply)
                    st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                    st.session_state.sent = True
                    st.rerun()
            else:
                st.session_state.sent = False

    # ==========================
    # End Encounter + Download
    # ==========================
    def build_transcript():
        buffer = io.StringIO()
        buffer.write(f"Doctor–Patient Conversation Log\nPatient: {st.session_state.patient_name}\n\n")
        for msg in st.session_state.history:
            speaker = "Doctor" if msg["role"] == "user" else "Patient"
            buffer.write(f"{speaker}: {msg['content']}\n")
        return buffer.getvalue().encode("utf-8")

    st.download_button(
        label="💾 End Encounter & Download",
        data=build_transcript(),
        file_name=f"encounter_{st.session_state.patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
