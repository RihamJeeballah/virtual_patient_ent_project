# app.py
import os, re, json, html, base64, tempfile, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS

# ==============================
# 🚀 APP & ENV SETUP
# ==============================
load_dotenv()
st.set_page_config(page_title="Virtual Patient", layout="centered")

# --- Required environment vars ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI_API_KEY in environment.")
    st.stop()

# --- Model & paths ---
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CASES_DIR = Path(os.getenv("CASES_DIR", "cases"))
AVATAR_DIR = Path(os.getenv("AVATAR_DIR", r"C:\Users\10User\Documents\virtual_patient_ent_project"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", "conversations")); LOGS_DIR.mkdir(exist_ok=True)

# --- Client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Feature flags (easy to toggle) ---
FEATURES = {
    "AUTO_PLAY_PATIENT_AUDIO": True,
    "AUTO_SEND_VOICE": True,
    "AUTO_SEND_TEXT_ON_ENTER": True,
    "LIMIT_HISTORY": 20,     # keep the last N turns to control token costs
    "EXPORT_DOWNLOAD": True
}

# ==============================
# 🔧 HELPERS
# ==============================
def esc(x: str) -> str:
    return html.escape(x).replace("\n", "<br>")

def pii_light_scrub(text: str) -> str:
    # Very light scrub for logs (emails/phones)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[email]", text)
    text = re.sub(r"\b\+?\d[\d\s\-\(\)]{7,}\b", "[phone]", text)
    return text

def save_log_line(session_id: str, role: str, content: str):
    log = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "role": role,
        "content": pii_light_scrub(content)
    }
    (LOGS_DIR / f"{session_id}.jsonl").write_text(
        ((LOGS_DIR / f"{session_id}.jsonl").read_text(encoding="utf-8") if (LOGS_DIR / f"{session_id}.jsonl").exists() else "")
        + json.dumps(log, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

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

def get_avatar_and_name(case_stem: str):
    # expects filenames like: blurred_eye_Salem.png → name="Salem"
    for f in AVATAR_DIR.glob(f"{case_stem}_*.png"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            patient_name = parts[-1].replace("-", " ").title()
            return str(f), patient_name
    return None, None

def build_patient_system_prompt(case: Dict) -> str:
    # Tight, commercialization-ready guardrails
    return (
        "You are role-playing a real human patient in a clinical interview.\n"
        "- Never reveal that you are simulated, an AI, or accessing a case file.\n"
        "- Answer ONLY what is asked. Do NOT volunteer extra details prematurely.\n"
        "- If the question is vague (e.g., 'what brings you here?'), state the chief complaint.\n"
        "- Maintain a realistic tone consistent with the scenario (pain/emotion if specified).\n"
        "- Do not suggest diagnoses or medical advice. You are the patient.\n"
        "- If the clinician asks something not in the case, say you don't remember or are unsure.\n\n"
        f"CASE (ground truth, do NOT quote this text):\n{json.dumps(case, ensure_ascii=False)}"
    )

def call_llm_as_patient(case: Dict, history: List[Dict[str, str]], temperature: float = 0.7) -> str:
    system = {"role": "system", "content": build_patient_system_prompt(case)}
    # Trim long histories to control token & latency
    trimmed = history[-FEATURES["LIMIT_HISTORY"]:] if FEATURES["LIMIT_HISTORY"] else history
    msgs = [system] + [{"role": m["role"], "content": m["content"]} for m in trimmed]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=300
        )
        reply = (resp.choices[0].message.content or "").strip()
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
        return (transcription.text or "").strip()
    except Exception as e:
        st.error(f"⚠️ Transcription failed: {e}")
        return ""

def render_audio_player_if_any(chat_html: str, audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    auto = "autoplay" if FEATURES["AUTO_PLAY_PATIENT_AUDIO"] else ""
    return chat_html + f"<audio {auto} controls><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"

# ==============================
# 🧠 STATE
# ==============================
if "session_id" not in st.session_state: st.session_state.session_id = uuid.uuid4().hex[:12]
if "case" not in st.session_state: st.session_state.case = None
if "case_name" not in st.session_state: st.session_state.case_name = ""
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None
if "history" not in st.session_state: st.session_state.history = []   # [{role, content, audio?}]
if "consented" not in st.session_state: st.session_state.consented = False
if "temperature" not in st.session_state: st.session_state.temperature = 0.7

# ==============================
# 🧱 HEADER / BRAND
# ==============================
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

st.title("💬 Virtual Patient — Text & Voice")

with st.expander("Privacy & Consent", expanded=not st.session_state.consented):
    st.markdown(
        "- Conversations may be logged for quality and research improvements.\n"
        "- Do not enter real patient identifiers.\n"
        "- By continuing, you consent to simulated interactions."
    )
    st.session_state.consented = st.checkbox("I understand and consent to proceed.", value=st.session_state.consented)
    if not st.session_state.consented:
        st.stop()

# ==============================
# 🎛️ CONTROLS (for testing / commercialization)
# ==============================
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    st.session_state.temperature = st.slider("Response variability", 0.0, 1.2, st.session_state.temperature, 0.1)
with c2:
    FEATURES["AUTO_PLAY_PATIENT_AUDIO"] = st.toggle("Auto-play patient voice", FEATURES["AUTO_PLAY_PATIENT_AUDIO"])
with c3:
    st.caption(f"Session: `{st.session_state.session_id}`")

# ==============================
# 📂 CASE SELECTION
# ==============================
if not st.session_state.case:
    st.subheader("Select a clinical case")
    cases = sorted(CASES_DIR.glob("*.md"))
    if not cases:
        st.warning("No case files found in the 'cases' folder.")
        st.stop()
    options = {f.stem.replace("_", " ").title(): f for f in cases}
    choice = st.selectbox("Case", ["— Select —"] + list(options.keys()))
    if choice != "— Select —":
        file_ = options[choice]
        st.session_state.case = load_case(file_)
        st.session_state.case_name = file_.stem
        st.session_state.avatar_path, st.session_state.patient_name = get_avatar_and_name(file_.stem)
        st.session_state.history = []
        st.success(f"✅ Loaded: {st.session_state.case['title']}")
        st.rerun()
else:
    case = st.session_state.case

    # ==============================
    # 🖼️ Patient banner
    # ==============================
    banner = st.container()
    with banner:
        bc1, bc2 = st.columns([1, 5])
        with bc1:
            if st.session_state.avatar_path:
                st.image(st.session_state.avatar_path, width=120, caption=st.session_state.patient_name or "")
        with bc2:
            st.markdown(
                f"<h2 style='margin:0'>{esc(st.session_state.patient_name or 'Patient')}</h2>"
                f"<p style='margin-top:4px;color:#666'>{esc(case.get('Setting',''))}</p>",
                unsafe_allow_html=True
            )

    # ==============================
    # 💬 Conversation (custom UI)
    # ==============================
    st.markdown("""
    <style>
      .chat {background:#F8F9FA;border-radius:14px;padding:10px;height:420px;overflow-y:auto;box-shadow:0 1px 10px rgba(0,0,0,0.06);}
      .bubble {padding:10px 14px;border-radius:18px;margin:6px;max-width:75%;font-size:15px;line-height:1.5}
      .doctor {background:#ffffff;text-align:left;border:1px solid #eee}
      .patient {background:#E7F5EE;text-align:left;align-self:flex-end;border:1px solid #d9f0e6}
      .role {font-weight:600;margin-bottom:4px}
      audio { width:100%; margin-top:6px }
    </style>
    """, unsafe_allow_html=True)

    chat_html = "<div class='chat' style='display:flex;flex-direction:column'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            chat_html += f"<div class='bubble doctor'><span class='role'>👨‍⚕️ Doctor:</span>{esc(m['content'])}</div>"
        else:
            chat_html += f"<div class='bubble patient'><span class='role'>🧑 Patient:</span>{esc(m['content'])}"
            if "audio" in m and m["audio"]:
                chat_html = render_audio_player_if_any(chat_html, m["audio"])
            chat_html += "</div>"
    chat_html += "<div id='bottom'></div></div>"
    chat_html += "<script>var c=document.querySelector('.chat'); if(c){c.scrollTo({top:c.scrollHeight, behavior:'smooth'});}</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ==============================
    # 🧩 Send functions
    # ==============================
    def handle_user_message(text: str, ui_suffix: str = ""):
        if not text or not text.strip():
            return
        content = text.strip()
        st.session_state.history.append({"role": "user", "content": content + ui_suffix})
        save_log_line(st.session_state.session_id, "user", content)
        reply = call_llm_as_patient(case, st.session_state.history, temperature=st.session_state.temperature)
        if reply:
            audio_path = tts_mp3(reply)
            st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_path})
            save_log_line(st.session_state.session_id, "assistant", reply)
        st.rerun()

    # ==============================
    # 🧑‍⚕️ Input row (Text + Voice)
    # ==============================
    st.subheader("Doctor Input")

    # -- Text: auto-send on Enter (modern chat UX)
    if FEATURES["AUTO_SEND_TEXT_ON_ENTER"]:
        user_text = st.chat_input("Type your question to the patient…")
        if user_text is not None:
            handle_user_message(user_text)
    else:
        with st.form("text_form", clear_on_submit=True):
            user_text = st.text_input("Type your question to the patient…")
            submitted = st.form_submit_button("Send")
            if submitted and user_text:
                handle_user_message(user_text)

    # -- Voice: auto-send after recording is captured
    audio_col = st.container()
    with audio_col:
        audio_data = st.audio_input("🎤 Record your question")
        if audio_data and FEATURES["AUTO_SEND_VOICE"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_data.read()); f.flush()
                text_transcribed = speech_to_text(f.name)
            if text_transcribed and len(text_transcribed) > 1:
                handle_user_message(text_transcribed, ui_suffix=" (🎤 Voice)")
            else:
                st.warning("⚠️ Voice message was too short or unclear. Please try again.")

    # ==============================
    # 📦 Export / Close
    # ==============================
    exp_c1, exp_c2 = st.columns([1,1])
    with exp_c1:
        if FEATURES["EXPORT_DOWNLOAD"]:
            if st.button("📥 Download conversation (JSONL)"):
                log_path = LOGS_DIR / f"{st.session_state.session_id}.jsonl"
                if log_path.exists():
                    st.download_button(
                        "Download now",
                        data=log_path.read_bytes(),
                        file_name=f"{st.session_state.case_name}_{st.session_state.session_id}.jsonl",
                        mime="application/json"
                    )
                else:
                    st.info("No log yet for this session.")
    with exp_c2:
        if st.button("End Encounter"):
            # Lightweight human-readable summary package
            summary = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history])
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.md"
            st.download_button("📥 Download Summary", summary, file_name=fname)
            # Reset (keep session id)
            st.session_state.case = None
            st.session_state.history = []
            st.experimental_rerun()
