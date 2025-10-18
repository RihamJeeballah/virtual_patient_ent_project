import os, re, json, html, base64, tempfile
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
# 🏫 HEADER BANNER (LEFT ALIGNED + SHADED BACKGROUND)
# ==========================
LOGO_PATH = "logo.png"

st.markdown(f"""
<style>
div[data-testid="stDecoration"] {{ display: none; }}
.block-container {{ padding-top: 0rem; }}

/* Banner Container */
.header-banner {{
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background: linear-gradient(90deg, #f0f2f5 0%, #e6e9ef 100%);
    display: flex;
    justify-content: flex-start;
    align-items: center;
    padding: 25px 60px;
    box-sizing: border-box;
    border-bottom: 1px solid #d0d5dd;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
}}

/* Logo + Text Container */
.header-content {{
    display: flex;
    align-items: center;
    gap: 20px;
}}

.header-banner img {{
    height: 90px;
}}

.header-text {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: left;
}}

.header-text h1 {{
    font-size: 28px;
    font-weight: 700;
    color: #222;
    margin: 0;
}}

.header-text h2 {{
    font-size: 20px;
    font-weight: 500;
    color: #333;
    margin: 0;
}}

.header-text h3 {{
    font-size: 16px;
    font-weight: 400;
    color: #555;
    margin: 0;
}}
</style>

<div class='header-banner'>
  <div class='header-content'>
    <img src='data:image/png;base64,{base64.b64encode(open(LOGO_PATH, "rb").read()).decode()}' alt='Logo'>
    <div class='header-text'>
        <h1>Sultan Qaboos University</h1>
        <h2>College of Medicine and Health Sciences</h2>
        <h3>Clinical Skills Lab</h3>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# ✨ STYLING
# ==========================
st.markdown("""
<style>
body, .block-container {background-color: #f8f9fb;}
h1,h2,h3 {font-family: 'Segoe UI', sans-serif;}
.avatar-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    background: white;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 20px;
    transition: box-shadow 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    cursor: pointer;
}
.avatar-card:hover {
    box-shadow: 0 4px 14px rgba(0,0,0,0.1);
}
.avatar-card img {
    border-radius: 50%;
    width: 130px;
    height: 130px;
    object-fit: cover;
}
.avatar-name { font-weight: 700; margin-top: 8px; font-size: 16px; color: #333; }
.avatar-case { color: #666; font-size: 14px; }
.chat-header {
    display: flex;
    align-items: center;
    background: white;
    border-radius: 14px;
    padding: 10px 20px;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.chat-header img {
    border-radius: 50%;
    width: 60px;
    height: 60px;
    object-fit: cover;
    margin-right: 15px;
}
.chat {
    background:#FFFFFF;
    border-radius:14px;
    padding:15px;
    height: 500px;
    overflow-y:auto;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}
.bubble {
    padding:10px 14px;
    border-radius:18px;
    margin:8px;
    max-width:70%;
    font-size:15px;
    line-height:1.5;
}
.doctor { background:#eef1f5; text-align:left; align-self:flex-start; }
.patient { background:#e7f5ee; text-align:left; align-self:flex-end; }
.role { font-weight:600; margin-bottom:4px; }
.chat-footer {
    position: sticky;
    bottom: 0;
    background: #f8f9fb;
    padding-top: 10px;
}
.stButton>button {
    background-color: #4B72FF;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 6px;
}
.stButton>button:hover {
    background-color: #3a5bd1;
}
[data-baseweb="input"] input {
    border-radius: 8px !important;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# 🧠 HELPERS
# ==========================
def esc(x: str) -> str:
    return html.escape(x).replace("\n", "<br>")

def load_case(file_path: Path) -> Dict[str, str]:
    text = file_path.read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)
    case = {"title": sections[0].strip("# \n")}
    for sec in sections[1:]:
        parts = sec.split("\n", 1)
        header = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        case[header] = body
    return case

def match_case_by_name(case_name: str):
    for cf in CASES_DIR.glob("*.md"):
        if case_name.lower() in cf.stem.lower():
            return cf
    return None

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
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=0.8,
            max_tokens=300
        )
        reply = resp.choices[0].message.content.strip()
        return reply or "…(The patient seems unsure and stays silent.)"
    except Exception as e:
        st.error(f"❌ LLM request failed: {e}")
        return "…(The patient is silent.)"

def tts_mp3(text: str) -> str:
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

def speech_to_text(audio_file) -> str:
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

# ==========================
# STATE
# ==========================
if "case" not in st.session_state: st.session_state.case = None
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None
if "case_name" not in st.session_state: st.session_state.case_name = None
if "history" not in st.session_state: st.session_state.history = []

# ==========================
# 🧍 PATIENT SELECTION PAGE
# ==========================
if not st.session_state.case:
    st.subheader("🩺 Select a Patient Case")

    avatars = [a for a in sorted(AVATAR_DIR.glob("*.png")) if a.stem.lower() != "logo"]
    num_cols = 4
    cols = st.columns(num_cols)

    for i, avatar in enumerate(avatars):
        parts = avatar.stem.split("_")
        case_name = " ".join(parts[:-1]).title()
        patient_name = parts[-1].title()
        col = cols[i % num_cols]
        with col:
            if st.button(f"🧑 {patient_name}\n🩺 {case_name}", key=f"btn_{avatar.stem}"):
                matched_case = match_case_by_name("_".join(parts[:-1]))
                if matched_case:
                    st.session_state.case = load_case(matched_case)
                    st.session_state.case_name = matched_case.stem
                    st.session_state.avatar_path = str(avatar)
                    st.session_state.patient_name = patient_name
                    st.session_state.history = []
                    st.rerun()
            st.markdown(f"""
                <div class='avatar-card'>
                    <img src='data:image/png;base64,{base64.b64encode(open(str(avatar), "rb").read()).decode()}'>
                    <div class='avatar-name'>{patient_name}</div>
                    <div class='avatar-case'>{case_name}</div>
                </div>
            """, unsafe_allow_html=True)

# ==========================
# 💬 CHAT PAGE
# ==========================
else:
    case = st.session_state.case

    # Header
    st.markdown(f"""
    <div class='chat-header'>
        <img src='data:image/png;base64,{base64.b64encode(open(st.session_state.avatar_path, "rb").read()).decode()}'>
        <div>
            <h3 style='margin:0'>{st.session_state.patient_name}</h3>
            <div style='color:#777;font-size:14px;'>{case.get("title","")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Back to Patients"):
        st.session_state.case = None
        st.session_state.history = []
        st.rerun()

    # Chat area
    chat_html = "<div class='chat' style='display:flex;flex-direction:column'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            chat_html += f"<div class='bubble doctor'><span class='role'>👨‍⚕️</span>{esc(m['content'])}</div>"
        else:
            chat_html += f"<div class='bubble patient'><span class='role'>🧑</span>{esc(m['content'])}"
            if "audio" in m:
                with open(m["audio"], "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chat_html += f"<audio autoplay><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
            chat_html += "</div>"
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var c=document.querySelector('.chat'); if(c){c.scrollTo({top:c.scrollHeight, behavior:'smooth'});}</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Footer input
    with st.container():
        user_text = st.chat_input("Type your question to the patient…")
        if user_text:
            st.session_state.history.append({"role": "user", "content": user_text})
            reply = call_llm_as_patient(case, st.session_state.history)
            if reply:
                audio_path = tts_mp3(reply)
                st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
            st.rerun()

        audio_data = st.audio_input("🎤 Record your question")
        if audio_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data.read())
                f.flush()
                text_transcribed = speech_to_text(f.name)

            if text_transcribed and len(text_transcribed) > 2:
                st.session_state.history.append({"role": "user", "content": text_transcribed + " (🎤 Voice)"})
                reply = call_llm_as_patient(case, st.session_state.history)
                if reply:
                    audio_path = tts_mp3(reply)
                    st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
                st.rerun()
            else:
                st.warning("⚠️ Voice message was too short or unclear.")
