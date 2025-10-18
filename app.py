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
AVATAR_DIR = Path(r"C:\Users\10User\Documents\virtual_patient_ent_project")
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

def get_avatar_and_name(case_stem: str):
    """Get avatar path and patient name based on file naming pattern."""
    for f in AVATAR_DIR.glob(f"{case_stem}_*.png"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            patient_name = parts[-1].capitalize()
            return str(f), patient_name
    return None, None

def call_llm_as_patient(case: Dict, history: List[Dict[str, str]]) -> str:
    """Call LLM to simulate patient response with strong patient instructions"""
    system = {
        "role": "system",
        "content": (
            "You are role-playing a human patient in a medical interview. "
            "Never say you're an AI or simulator. "
            "Do not reveal everything at once — answer only what was asked. "
            "If the question is vague, reply with your main symptom. "
            "Keep tone realistic and human-like.\n\n"
            f"CASE:\n{json.dumps(case)}"
        )
    }

    msgs = [system] + [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.8,
            max_tokens=300
        )
        reply = resp.choices[0].message.content.strip()
        if not reply:
            reply = "…(The patient seems unsure and stays silent.)"
        return reply
    except Exception as e:
        st.error(f"❌ LLM request failed: {e}")
        return "…(The patient is silent.)"

def tts_mp3(text: str) -> str:
    """Convert text to speech and save as mp3"""
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def speech_to_text(audio_file) -> str:
    """Transcribe recorded audio using Whisper with error handling"""
    try:
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcription.text.strip()
    except Exception as e:
        st.error(f"⚠️ Transcription failed: {e}")
        return ""

def esc(x: str) -> str:
    return html.escape(x).replace("\n", "<br>")

# ------------------ STATE ------------------
if "case" not in st.session_state: st.session_state.case = None
if "history" not in st.session_state: st.session_state.history = []
if "started" not in st.session_state: st.session_state.started = False
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None

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
            selected_file = case_files[choice]
            st.session_state.case = load_case(selected_file)
            st.session_state.case_name = selected_file.stem
            st.session_state.avatar_path, st.session_state.patient_name = get_avatar_and_name(selected_file.stem)
            st.session_state.history = []
            st.session_state.started = False
            st.success(f"✅ Loaded case: {st.session_state.case['title']}")
else:
    case = st.session_state.case

    # ------------------ Patient Avatar ------------------
    if st.session_state.avatar_path:
        colA, colB = st.columns([1, 4])
        with colA:
            st.image(st.session_state.avatar_path, width=120)
        with colB:
            st.markdown(f"<h2 style='margin:0'>{st.session_state.patient_name}</h2>", unsafe_allow_html=True)

    # ------------------ Chat Display ------------------
    st.markdown("""
    <style>
    .chat {background:#f6f6f6;border-radius:14px;padding:10px;height:420px;overflow-y:auto;box-shadow:0 0 8px rgba(0,0,0,0.1);}
    .bubble {padding:10px 14px;border-radius:18px;margin:6px;max-width:75%;font-size:15px;line-height:1.4}
    .doctor {background:#fff;text-align:left}
    .patient {background:#e6f4ea;text-align:left;align-self:flex-end}
    .role {font-weight:600;margin-bottom:4px}
    </style>
    """, unsafe_allow_html=True)

    chat_html = "<div class='chat' style='display:flex;flex-direction:column'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            suffix = m.get("ui_suffix", "")
            chat_html += f"<div class='bubble doctor'><span class='role'>👨‍⚕️ Doctor:</span>{esc(m['content'])}{esc(suffix)}</div>"
        else:
            chat_html += f"<div class='bubble patient'><span class='role'>🧑 Patient:</span>{esc(m['content'])}"
            if "audio" in m:
                with open(m["audio"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chat_html += f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
            chat_html += "</div>"
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var c=document.querySelector('.chat'); if(c){c.scrollTo({top:c.scrollHeight, behavior:'smooth'});}</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ------------------ DOCTOR INPUT ------------------
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
                if reply.strip():
                    audio_path = tts_mp3(reply)
                    st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                st.rerun()

    # Voice input — AUTO SEND ON RECORD FINISH
    with tab2:
        audio_data = st.audio_input("Record your message:")
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data.read())
                f.flush()
                text_transcribed = speech_to_text(f.name)

            if text_transcribed and len(text_transcribed) > 2:
                st.session_state.history.append({"role": "user", "content": text_transcribed.strip(), "ui_suffix": " (🎤 Voice)"})
                st.session_state.started = True
                reply = call_llm_as_patient(case, st.session_state.history)
                if reply.strip():
                    audio_path = tts_mp3(reply)
                    st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                st.rerun()
            else:
                st.warning("⚠️ Voice message was too short or unclear. Please try again.")

    # ------------------ END ENCOUNTER ------------------
    if st.button("End Encounter"):
        summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.md"
        st.download_button("📥 Download Summary", summary, file_name=fname)
        st.session_state.case = None
