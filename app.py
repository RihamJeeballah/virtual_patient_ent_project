import streamlit as st
import os
import re
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS

# ======== LOAD API KEY ========
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ======== CONFIG ========
CASES_DIR = Path("cases")
LOGS_DIR = Path("conversations")
LOGS_DIR.mkdir(exist_ok=True)
MODEL = "gpt-4o-mini"

# ======== HELPERS ========
def load_case(file_path: Path) -> Dict[str, str]:
    text = file_path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)
    case = {}
    case["title"] = sections[0].strip("# \n")
    for sec in sections[1:]:
        parts = sec.split("\n", 1)
        header = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        case[header] = body
    return case

def list_cases():
    return [f for f in CASES_DIR.glob("*.md")]

def call_llm(prompt: str, history: list):
    try:
        messages = [{"role": "system", "content": prompt}] + history
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"⚠️ LLM error: {e}")
        return "⚠️ Sorry, I couldn’t generate a response."

def summarize_chat(history):
    text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])
    return f"# Encounter Summary\n\n{text}"

def tts_response(text):
    """Convert patient reply to audio using gTTS"""
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def transcribe_audio(file_path):
    """Use Whisper STT to transcribe doctor's audio input"""
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# ======== STREAMLIT UI ========
st.set_page_config(page_title="Virtual ENT Patient", layout="centered")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpg", width=110)
with col2:
    st.markdown(
        """
        <div style="text-align: left;">
            <h1 style="margin-bottom: 0;">Sultan Qaboos University</h1>
            <h2 style="margin-bottom: 0;">College of Medicine</h2>
            <h3 style="color: gray; margin-top: 0;">Clinical Skills Lab</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.title("🩺 Virtual ENT Patient (Voice + Text)")

# ======== STATE ========
if "case" not in st.session_state:
    st.session_state.case = None
if "history" not in st.session_state:
    st.session_state.history = []
if "started" not in st.session_state:
    st.session_state.started = False

# ======== CASE SELECTION ========
if not st.session_state.case:
    st.header("Select a clinical case:")
    cases = list_cases()
    case_files = {f.stem.replace("_", " ").title(): f for f in cases}
    choice = st.selectbox("Choose a case", ["--Select--"] + list(case_files.keys()))
    if choice != "--Select--":
        file_path = case_files[choice]
        case_data = load_case(file_path)
        st.session_state.case = case_data
        st.session_state.case_name = file_path.stem
        st.session_state.history = []
        st.session_state.started = False
        st.success(f"Loaded case: {case_data['title']}")

else:
    case = st.session_state.case
    st.subheader(case["title"])
    st.markdown(f"**Opening Stem:** {case.get('Opening Stem','')}")

    # ======== Initial greeting ========
    if not st.session_state.started:
        greeting = "Hello, I'm your doctor today. What brings you in?"
        st.session_state.history.append({"role": "user", "content": greeting})

        patient_prompt = f"""
You are the patient in this ENT clinical case.
Respond naturally, using first-person language, without giving away all details at once.
Tone: match the scenario's tone.

Case details:
{json.dumps(case, indent=2)}
"""
        patient_response = call_llm(patient_prompt, [{"role": "user", "content": greeting}])
        audio_path = tts_response(patient_response)
        st.session_state.history.append({"role": "assistant", "content": patient_response, "audio": audio_path})
        st.session_state.started = True

    # ======== Chat UI ========
    st.markdown(
        """
        <style>
        .chat-container { height: 500px; overflow-y: auto; display: flex; flex-direction: column; }
        .chat-bubble { border-radius: 15px; padding: 10px 15px; margin: 5px 0; max-width: 80%; word-wrap: break-word; }
        .doctor { background-color: #D7EBFF; color: black; align-self: flex-start; }
        .patient { background-color: #F1F1F1; color: black; align-self: flex-end; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_html = "<div class='chat-container'>"
    for msg in st.session_state.history:
        if msg["role"] == "user":
            chat_html += f"<div class='chat-bubble doctor'>👨‍⚕️ <strong>Doctor:</strong> {msg['content']}</div>"
        else:
            chat_html += f"<div class='chat-bubble patient'>🧑 <strong>Patient:</strong> {msg['content']}</div>"
            if "audio" in msg:
                st.audio(msg["audio"], format="audio/mp3")
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var chat=document.querySelector('.chat-container');chat.scrollTop=chat.scrollHeight;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ======== TEXT INPUT ========
    if prompt := st.chat_input("Ask your patient..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        system_prompt = f"""
You are simulating the patient described in the following clinical case.
- Speak in first person only.
- Do not give all details at once.
- Do not use information outside this case.
- Match tone to the scenario.

Case details:
{json.dumps(case, indent=2)}
"""
        reply = call_llm(system_prompt, st.session_state.history)
        audio_path = tts_response(reply)
        st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
        st.rerun()

    # ======== VOICE INPUT ========
    st.markdown("🎤 **Record your question:**")
    audio_file = st.audio_input("Record voice")
    if audio_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_file.getvalue())
        temp_file.flush()
        transcription = transcribe_audio(temp_file.name)
        st.session_state.history.append({"role": "user", "content": transcription})
        system_prompt = f"""
You are simulating the patient described in the following clinical case.
- Speak in first person only.
- Do not give all details at once.
- Do not use information outside this case.
- Match tone to the scenario.

Case details:
{json.dumps(case, indent=2)}
"""
        reply = call_llm(system_prompt, st.session_state.history)
        audio_path = tts_response(reply)
        st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
        st.rerun()

    # ======== END ENCOUNTER ========
    if st.button("End Encounter"):
        summary = summarize_chat(st.session_state.history)
        st.session_state.summary = summary
        log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.json"
        with open(LOGS_DIR / log_name, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, indent=2)
        st.markdown("### 📝 Encounter Summary")
        st.markdown(summary)
        st.download_button("Download Summary (.md)", summary, file_name="summary.md")
        st.session_state.case = None
