import streamlit as st
import os
import re
import json
import tempfile
import base64
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
        # Strip audio and UI fields before sending to LLM
        clean_history = [{"role": h["role"], "content": h["content"]} for h in history]
        messages = [{"role": "system", "content": prompt}] + clean_history
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
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcript.text

# ======== UI ========
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

    # ======== First patient response ========
    if not st.session_state.started:
        greeting = "Hello, I'm your doctor today. What brings you in?"
        st.session_state.history.append({"role": "user", "content": greeting})

        patient_prompt = f"""
You are role-playing as a human patient in a clinical encounter. 
Strictly follow these rules:

1. Act and speak as a real patient would. 
   - Use **first person language only** (e.g., “I have pain in my right ear” not “The patient has pain…”).
   - Do not use medical jargon or structured summaries.
   - Respond naturally and conversationally.

2. Only disclose information if asked directly or if it’s natural for a patient to volunteer it at that point. 
   - Do not give all details at once.
   - If the doctor’s question is vague or partial, give a **limited but natural** patient-like answer.
   - For example, if asked “Tell me more,” give a brief but realistic description, not the entire HPI.

3. Do **not** learn from or reference any previous conversations or outside knowledge. 
   Use **only the information contained in this case file** below.
   If the doctor asks something unrelated to the case, politely say you don't know or don't remember.

4. Adjust your tone and style to match the scenario:
   - Reflect the patient’s level of pain, discomfort, or anxiety.
   - Speak with emotion or hesitation where appropriate (e.g., “It really hurts when I touch it.”).

5. Stay in character at all times as the patient.

Background case details:
{json.dumps(case, indent=2)}
"""

        patient_response = call_llm(patient_prompt, [{"role": "user", "content": greeting}])
        audio_path = tts_response(patient_response)
        st.session_state.history.append({"role": "assistant", "content": patient_response, "audio": audio_path})
        st.session_state.started = True

    # ======== CHAT UI ========
    st.markdown(
        """
        <style>
        .chat-container {
            height: 520px;
            overflow-y: auto;
            padding: 10px;
            background-color: #FAFAFA;
            border-radius: 15px;
            border: 1px solid #DDD;
            display: flex;
            flex-direction: column;
        }
        .chat-bubble {
            border-radius: 18px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            line-height: 1.4;
        }
        .doctor {
            background-color: #D6EBFF;
            color: #000;
            align-self: flex-start;
            text-align: left;
        }
        .patient {
            background-color: #EDEDED;
            color: #000;
            align-self: flex-end;
            text-align: left;
        }
        .chat-role {
            font-weight: bold;
            display: block;
            margin-bottom: 4px;
        }
        audio {
            width: 100%;
            margin-top: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_html = "<div class='chat-container'>"

    for msg in st.session_state.history:
        if msg["role"] == "user":
            icon = "🎧 " if msg.get("recorded") else ""
            chat_html += f"""
            <div class='chat-bubble doctor'>
                <span class='chat-role'>👨‍⚕️ Doctor:</span>
                {icon}{msg['content']}
            </div>
            """
        else:
            chat_html += f"""
            <div class='chat-bubble patient'>
                <span class='chat-role'>🧑 Patient:</span>
                {msg['content']}
            """
            if "audio" in msg:
                with open(msg["audio"], "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                chat_html += f"""
                <audio controls>
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                """
            chat_html += "</div>"

    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var chat=document.querySelector('.chat-container');chat.scrollTop=chat.scrollHeight;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ======== TEXT INPUT ========
    if prompt := st.chat_input("Ask your patient..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        system_prompt = f"""
You are role-playing as a human patient in a clinical encounter.
- Speak in first person only.
- Do not give all details at once.
- Do not use external knowledge.
- Reflect scenario tone and pain level.

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
        st.session_state.history.append({"role": "user", "content": transcription, "recorded": True})

        system_prompt = f"""
You are role-playing as a human patient in a clinical encounter.
- Speak in first person only.
- Do not give all details at once.
- Use only the information from this case.

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
