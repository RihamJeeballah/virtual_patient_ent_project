import os, re, json, html, base64, tempfile, uuid
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
AVATAR_DIR = Path(".")  # same folder where images are stored
LOGS_DIR = Path("conversations")
LOGS_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# ✨ CSS
# ==========================
st.markdown("""
<style>
/* General layout */
.chat {
  background:#F8F9FA;
  border-radius:14px;
  padding:10px;
  height:480px;
  overflow-y:auto;
  box-shadow:0 2px 10px rgba(0,0,0,0.05);
}
.bubble {
  padding:10px 14px;
  border-radius:18px;
  margin:6px;
  max-width:75%;
  font-size:15px;
  line-height:1.5;
}
.doctor {
  background:#ffffff;
  text-align:left;
  border:1px solid #eee;
}
.patient {
  background:#E7F5EE;
  text-align:left;
  align-self:flex-end;
  border:1px solid #d9f0e6;
}
.role { font-weight:600; margin-bottom:4px }
audio { width:100%; margin-top:6px }
.stButton>button {white-space:normal;height:auto;}
.avatar-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #ffffff;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 10px;
  margin-bottom: 15px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
  cursor: pointer;
  transition: all 0.2s ease;
}
.avatar-btn:hover {
  box-shadow: 0 3px 10px rgba(0,0,0,0.1);
  background: #f8f8f8;
}
.avatar-name { font-weight: 600; margin-top: 5px; font-size: 15px; }
.avatar-case { color: #666; font-size: 13px; }
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
    # Match avatar file name (e.g. blocked_nose) with case markdown file
    for cf in CASES_DIR.glob("*.md"):
        if case_name.lower() in cf.stem.lower():
            return cf
    return None

def call_llm_as_patient(case: Dict, history: List[Dict[str, str]]) -> str:
    system = {
        "role": "system",
        "content": (
            "You are role-playing a real human patient in a clinical interview.\n"
            "- Never reveal that you are simulated.\n"
            "- Respond naturally and only to what was asked.\n"
            "- If the question is vague, give your main symptom.\n"
            "- Do not give the entire history at once.\n"
            "- Keep tone human-like and realistic.\n\n"
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
# 🧭 STATE
# ==========================
if "case" not in st.session_state: st.session_state.case = None
if "case_name" not in st.session_state: st.session_state.case_name = None
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None
if "history" not in st.session_state: st.session_state.history = []

# ==========================
# 🧍 AVATAR CASE SELECTION
# ==========================
if not st.session_state.case:
    st.subheader("Select a Patient Case")
    avatars = sorted(AVATAR_DIR.glob("*.png"))
    num_cols = 3
    cols = st.columns(num_cols)

    for i, avatar in enumerate(avatars):
        parts = avatar.stem.split("_")
        case_name = " ".join(parts[:-1]).title()
        patient_name = parts[-1].title()
        col = cols[i % num_cols]
        with col:
            with st.container():
                st.image(str(avatar), width=150)
                if st.button(f"🧑 {patient_name}\n🩺 {case_name}", key=f"avatar_{avatar.stem}", use_container_width=True):
                    matched_case = match_case_by_name("_".join(parts[:-1]))
                    if matched_case:
                        st.session_state.case = load_case(matched_case)
                        st.session_state.case_name = matched_case.stem
                        st.session_state.avatar_path = str(avatar)
                        st.session_state.patient_name = patient_name
                        st.session_state.history = []
                        st.rerun()

# ==========================
# 💬 CHAT INTERFACE
# ==========================
else:
    case = st.session_state.case

    # Back button
    if st.button("⬅️ Back to Patients"):
        st.session_state.case = None
        st.session_state.history = []
        st.rerun()

    # Patient Banner
    colA, colB = st.columns([1, 5])
    with colA:
        st.image(st.session_state.avatar_path, width=130)
    with colB:
        st.markdown(f"<h2 style='margin:0'>{st.session_state.patient_name}</h2><p style='color:#666'>{esc(case.get('Setting',''))}</p>", unsafe_allow_html=True)

    # Chat display
    chat_html = "<div class='chat' style='display:flex;flex-direction:column'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            chat_html += f"<div class='bubble doctor'><span class='role'>👨‍⚕️ Doctor:</span>{esc(m['content'])}</div>"
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

    # Text input (auto-send)
    user_text = st.chat_input("Type your question to the patient…")
    if user_text:
        st.session_state.history.append({"role": "user", "content": user_text})
        reply = call_llm_as_patient(case, st.session_state.history)
        if reply:
            audio_path = tts_mp3(reply)
            st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
        st.rerun()

    # Voice input (auto-send after recording)
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
            st.warning("⚠️ Voice message was too short or unclear. Please try again.")
