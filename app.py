import streamlit as st
import streamlit.components.v1 as components
import base64, tempfile, os, re, json, html
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS

# ------------------ SETUP ------------------
load_dotenv()
st.set_page_config(page_title="Virtual Patient Chat", layout="centered")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"
CASES_DIR = Path("cases")
LOGS_DIR = Path("conversations")
LOGS_DIR.mkdir(exist_ok=True)

# ------------------ HELPERS ------------------
def load_case(file_path: Path) -> Dict[str, str]:
    """Load markdown case into structured dict"""
    text = file_path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)
    case = {"title": sections[0].strip("# \n")}
    for sec in sections[1:]:
        parts = sec.split("\n", 1)
        header = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        case[header] = body
    return case

def call_llm_as_patient(case: Dict, history: List[Dict[str, str]]) -> str:
    """Call LLM to simulate patient response"""
    system = {
        "role": "system",
        "content": (
            "You are role-playing a human patient. "
            "Use ONLY the information in the case. "
            "Speak in first person. Reveal details gradually. "
            "Reflect tone/pain from scenario. Stay in character.\n\n"
            f"CASE:\n{json.dumps(case)}"
        )
    }
    msgs = [system] + history
    resp = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=0.8,
        max_tokens=300
    )
    return resp.choices[0].message.content

def tts_mp3(text: str) -> str:
    """Convert text to speech and save as mp3"""
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def speech_to_text(audio_file) -> str:
    """Transcribe recorded audio using Whisper"""
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcription.text.strip()

def esc(x: str) -> str:
    return html.escape(x).replace("\n", "<br>")

# ------------------ STATE ------------------
if "case" not in st.session_state: st.session_state.case = None
if "history" not in st.session_state: st.session_state.history = []
if "started" not in st.session_state: st.session_state.started = False

# ------------------ HEADER ------------------
col1, col2 = st.columns([1, 5])
with col1:
    if Path("logo.jpg").exists():
        st.image("logo.jpg", width=110)
with col2:
    st.markdown("""
        <div style="text-align:left">
            <h1 style="margin:0">Sultan Qaboos University</h1>
            <h2 style="margin:0">College of Medicine</h2>
            <h3 style="color:gray;margin-top:0">Clinical Skills Lab</h3>
        </div>
    """, unsafe_allow_html=True)

st.title("💬 Virtual Patient (Text & Voice)")

# ------------------ CASE SELECTION ------------------
if not st.session_state.case:
    st.header("Select a clinical case:")
    cases = list(CASES_DIR.glob("*.md"))
    if not cases:
        st.warning("No case files found in 'cases' folder.")
    else:
        case_files = {f.stem.replace("_", " ").title(): f for f in cases}
        choice = st.selectbox("Choose a case", ["--Select--"] + list(case_files.keys()))
        if choice != "--Select--":
            st.session_state.case = load_case(case_files[choice])
            st.session_state.case_name = case_files[choice].stem
            st.session_state.history = []
            st.session_state.started = False
            st.success(f"✅ Loaded case: {st.session_state.case['title']}")
else:
    case = st.session_state.case
    st.subheader(case["title"])

    # ------------------ Chat Display ------------------
    st.markdown("""
    <style>
    .chat {background:#ECE5DD;border-radius:14px;padding:10px;height:420px;overflow:auto}
    .bubble {padding:10px 14px;border-radius:18px;margin:6px;max-width:75%;font-size:15px;line-height:1.4}
    .doctor {background:#fff;text-align:left}
    .patient {background:#DCF8C6;text-align:left;align-self:flex-end}
    .role {font-weight:600;margin-bottom:4px}
    audio {width:100%;margin-top:6px}
    </style>
    """, unsafe_allow_html=True)

    chat_html = "<div class='chat' style='display:flex;flex-direction:column'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            chat_html += f"<div class='bubble doctor'><span class='role'>👨‍⚕️ Doctor:</span>{esc(m['content'])}</div>"
        else:
            chat_html += f"<div class='bubble patient'><span class='role'>🧑 Patient:</span>{esc(m['content'])}"
            if "audio" in m:
                with open(m["audio"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chat_html += f"<audio controls><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
            chat_html += "</div>"
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var c=document.querySelector('.chat'); c.scrollTop=c.scrollHeight;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ------------------ DOCTOR INPUT ------------------
    if not st.session_state.started and len(st.session_state.history) == 0:
        st.info("👩‍⚕️ Start the conversation by typing or recording your question.")

    st.markdown("### 🩺 Doctor Input Options")
    tab1, tab2 = st.tabs(["✍️ Text", "🎤 Voice"])

    # Text input
    with tab1:
        user_input = st.text_input("Type your message:", key="chat_input")
        if st.button("Send Text"):
            if user_input.strip():
                st.session_state.history.append({"role": "user", "content": user_input.strip()})
                st.session_state.started = True
                reply = call_llm_as_patient(case, st.session_state.history)
                audio_path = tts_mp3(reply)
                st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                st.rerun()

    # Voice input
    with tab2:
        audio_data = st.audio_input("Record your message:")
        if audio_data and st.button("Send Voice"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data.read())
                f.flush()
                text_transcribed = speech_to_text(f.name)
            st.session_state.history.append({"role": "user", "content": text_transcribed + " (🎤 Voice)"})
            st.session_state.started = True
            reply = call_llm_as_patient(case, st.session_state.history)
            audio_path = tts_mp3(reply)
            st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
            st.rerun()

    # ------------------ END ENCOUNTER ------------------
    if st.button("End Encounter"):
        summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.md"
        st.download_button("📥 Download Summary", summary, file_name=fname)
        st.session_state.case = None
