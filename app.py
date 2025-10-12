import streamlit as st
import os, re, json, tempfile, base64, html
from datetime import datetime
from pathlib import Path
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv
from gtts import gTTS

# ======== SETUP ========
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

def escape_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")

def tts_response(text):
    tts = gTTS(text=text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcript.text

def call_llm(prompt: str, history: list):
    clean_history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": prompt}] + clean_history
    completion = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.8, max_tokens=300
    )
    return completion.choices[0].message.content

def add_message(role, content, **kwargs):
    msg_id = f"{role}-{hash(content)}"
    if not any(m.get("id") == msg_id for m in st.session_state.history):
        st.session_state.history.append({"role": role, "content": content, "id": msg_id, **kwargs})

# ======== PAGE CONFIG ========
st.set_page_config(page_title="Virtual ENT Patient", layout="centered")

# ======== HEADER ========
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpg", width=110)
with col2:
    st.markdown("""
        <div style="text-align: left;">
            <h1 style="margin-bottom: 0;">Sultan Qaboos University</h1>
            <h2 style="margin-bottom: 0;">College of Medicine</h2>
            <h3 style="color: gray; margin-top: 0;">Clinical Skills Lab</h3>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.title("🩺 Virtual ENT Patient (Voice + Text)")

# ======== STATE ========
if "case" not in st.session_state: st.session_state.case = None
if "history" not in st.session_state: st.session_state.history = []
if "started" not in st.session_state: st.session_state.started = False
if "recording_preview" not in st.session_state: st.session_state.recording_preview = ""

# ======== CASE SELECTION ========
if not st.session_state.case:
    st.header("Select a clinical case:")
    cases = [f for f in CASES_DIR.glob("*.md")]
    case_files = {f.stem.replace("_", " ").title(): f for f in cases}
    choice = st.selectbox("Choose a case", ["--Select--"] + list(case_files.keys()))
    if choice != "--Select--":
        file_path = case_files[choice]
        st.session_state.case = load_case(file_path)
        st.session_state.case_name = file_path.stem
        st.session_state.history = []
        st.session_state.started = False
        st.success(f"Loaded case: {st.session_state.case['title']}")

else:
    case = st.session_state.case
    st.subheader(case["title"])
    st.markdown(f"**Opening Stem:** {case.get('Opening Stem','')}")

    # ======== First greeting ========
    if not st.session_state.started:
        greeting = "Hello, I'm your doctor today. What brings you in?"
        add_message("user", greeting)
        reply = call_llm(f"You are the patient in this case:\n{json.dumps(case)}",
                         [{"role": "user", "content": greeting}])
        audio_path = tts_response(reply)
        add_message("assistant", reply, audio=audio_path)
        st.session_state.started = True

    # ======== CHAT UI ========
    st.markdown("""
        <style>
        .chat-container {height: 520px; overflow-y: auto; padding: 10px; background: #ECE5DD; border-radius: 15px;}
        .bubble {padding: 10px 14px; border-radius: 18px; margin: 6px; max-width: 75%; font-size: 15px; line-height: 1.4;}
        .doctor {background: #FFFFFF; align-self: flex-start; text-align: left;}
        .patient {background: #DCF8C6; align-self: flex-end; text-align: left;}
        .chat-role {font-weight: bold; display: block; margin-bottom: 4px;}
        audio {width: 100%; margin-top: 5px;}
        .mic-button {
            position: fixed; bottom: 25px; right: 25px;
            background-color: #25D366; color: white;
            font-size: 26px; padding: 18px;
            border-radius: 50%; cursor: pointer;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            z-index: 1000;
        }
        .mic-button:hover { background-color: #1DA955; }
        </style>
    """, unsafe_allow_html=True)

    chat_html = "<div class='chat-container' style='display:flex;flex-direction:column;'>"
    for msg in st.session_state.history:
        if msg["role"] == "user":
            icon = "🎧 " if msg.get("recorded") else ""
            chat_html += f"""
            <div class='bubble doctor'>
                <span class='chat-role'>👨‍⚕️ Doctor:</span>
                {icon}{escape_html(msg['content'])}
            </div>"""
        else:
            chat_html += f"""
            <div class='bubble patient'>
                <span class='chat-role'>🧑 Patient:</span>
                {escape_html(msg['content'])}"""
            if "audio" in msg:
                with open(msg["audio"], "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode()
                chat_html += f"""<audio controls>
                        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    </audio>"""
            chat_html += "</div>"
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var chat=document.querySelector('.chat-container');chat.scrollTop=chat.scrollHeight;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ======== Text Input ========
    if prompt := st.chat_input("Type your question..."):
        add_message("user", prompt)
        reply = call_llm(f"You are the patient in this case:\n{json.dumps(case)}", st.session_state.history)
        audio_path = tts_response(reply)
        add_message("assistant", reply, audio=audio_path)
        st.rerun()

    # ======== Floating Mic Button ========
    mic_clicked = st.button("🎤", key="mic_button", help="Hold to record", use_container_width=False)

    if mic_clicked:
        st.session_state.recording_preview = "🎙️ Recording... Release to send."
        st.toast("🎙️ Recording...")

    # ======== Hidden audio recorder ========
    if mic_clicked:
        audio_file = st.audio_input("hidden_audio", label_visibility="collapsed")
        if audio_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_file.getvalue())
            temp_file.flush()
            transcription = transcribe_audio(temp_file.name)
            st.session_state.recording_preview = ""
            add_message("user", transcription, recorded=True)
            reply = call_llm(f"You are the patient in this case:\n{json.dumps(case)}", st.session_state.history)
            audio_path = tts_response(reply)
            add_message("assistant", reply, audio=audio_path)
            st.rerun()

    # ======== END ENCOUNTER ========
    if st.button("End Encounter"):
        summary = "\n".join([f"{h['role']}: {h['content']}" for h in st.session_state.history])
        log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.json"
        with open(LOGS_DIR / log_name, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, indent=2)
        st.markdown("### 📝 Encounter Summary")
        st.markdown(summary)
        st.download_button("Download Summary (.md)", summary, file_name="summary.md")
        st.session_state.case = None
