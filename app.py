import os, re, json, html, base64, tempfile, io, threading
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
# 🏫 HEADER BANNER
# ==========================
LOGO_PATH = "logo.png"
st.markdown(f"""
<style>
div[data-testid="stDecoration"] {{ display: none; }}
header {{ display: none; }}
.block-container {{ padding-top: 0rem; margin-top: 0rem; }}
section.main {{ padding-top: 0rem; margin-top: 0rem; }}

.header-banner {{
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background: linear-gradient(90deg, #f0f2f5 0%, #e6e9ef 100%);
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    padding: 45px 60px 25px 60px;
    border-bottom: 1px solid #d0d5dd;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
    position: relative;
    z-index: 1000;
}}
.header-content {{ display: flex; align-items: center; gap: 20px; }}
.header-banner img {{ height: 90px; }}
.header-text {{ display: flex; flex-direction: column; justify-content: center; text-align: left; }}
.header-text h1 {{ font-size: 28px; font-weight: 700; color: #222; margin: 0; }}
.header-text h2 {{ font-size: 20px; font-weight: 500; color: #333; margin: 0; }}
.header-text h3 {{ font-size: 16px; font-weight: 400; color: #555; margin: 0; }}
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
.avatar-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.1); }
.avatar-card img {
    border-radius: 50%;
    width: 130px;
    height: 130px;
    object-fit: cover;
}
.avatar-name { font-weight: 700; margin-top: 8px; font-size: 16px; color: #333; }
.avatar-case { color: #666; font-size: 14px; }

.chat {
    background:#FFFFFF;
    border-radius:14px;
    height: 500px;
    overflow-y:auto;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
    display:flex;
    flex-direction: column;
    padding: 14px;
}
.bubble {
    padding:10px 14px;
    border-radius:18px;
    margin:8px;
    max-width:75%;
    font-size:15px;
    line-height:1.5;
    display: inline-block;
}
.doctor { background:#eef1f5; text-align:left; align-self:flex-start; }
.patient { background:#e7f5ee; text-align:left; align-self:flex-end; }

.chat-header-card {
    display:flex;flex-direction:column;align-items:center;gap:10px;
    background:white;padding:20px;border-radius:15px;
    box-shadow:0 2px 6px rgba(0,0,0,0.05);margin-bottom:15px;
}
.chat-header-card img {
    border-radius:15px;
    width:170px;height:170px;object-fit:cover;
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
    resp = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        temperature=0.8,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

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

# ==========================
# STATE
# ==========================
if "case" not in st.session_state: st.session_state.case = None
if "avatar_path" not in st.session_state: st.session_state.avatar_path = None
if "patient_name" not in st.session_state: st.session_state.patient_name = None
if "case_name" not in st.session_state: st.session_state.case_name = None
if "history" not in st.session_state: st.session_state.history = []
if "input_mode" not in st.session_state: st.session_state.input_mode = "keyboard"
if "is_replying" not in st.session_state: st.session_state.is_replying = False

# ==========================
# PATIENT SELECTION PAGE
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
# CHAT PAGE
# ==========================
# ==========================
# CHAT PAGE
# ==========================
else:
    st.button("⬅️ Back to Patients", on_click=lambda: (st.session_state.update({"case": None, "history": []}), st.rerun()))

    st.markdown(f"""
    <div class="chat-header-card">
        <img src='data:image/png;base64,{base64.b64encode(open(st.session_state.avatar_path, "rb").read()).decode()}'>
        <div style='text-align:center;'>
            <h2 style='margin:0'>{st.session_state.patient_name}</h2>
            <div style='color:#777;font-size:14px'>{st.session_state.case.get("title","")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with open(st.session_state.avatar_path, "rb") as img_f:
        patient_icon_b64 = base64.b64encode(img_f.read()).decode()

    chat_html = "<div class='chat' id='chatBox'>"
    for m in st.session_state.history:
        if m["role"] == "user":
            chat_html += f"<div class='bubble doctor'><span style='font-weight:600;margin-right:6px;'>👨‍⚕️</span>{esc(m['content'])}</div>"
        else:
            chat_html += f"""
            <div class='bubble patient'>
                <img src='data:image/png;base64,{patient_icon_b64}' style='width:26px;height:26px;border-radius:6px;vertical-align:middle;margin-right:8px;'>
                {esc(m['content'])}
            """
            if "audio" in m:
                try:
                    with open(m["audio"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    chat_html += f"<audio controls autoplay style='display:block;margin-top:4px;'><source src='data:audio/mp3;base64,{b64}' type='audio/mp3'></audio>"
                except FileNotFoundError:
                    pass
            chat_html += "</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ✅ Auto scroll always at bottom after rerun
    st.markdown("""
    <script>
    const chatBox = window.parent.document.getElementById('chatBox');
    if (chatBox) {
        chatBox.scrollTop = chatBox.scrollHeight;
        const observer = new MutationObserver(() => {
            chatBox.scrollTop = chatBox.scrollHeight;
        });
        observer.observe(chatBox, { childList: true, subtree: true });
    }
    </script>
    """, unsafe_allow_html=True)

    # 👇 Process pending message if there is one
    if "pending_message" in st.session_state and st.session_state.pending_message:
        user_msg = st.session_state.pending_message
        st.session_state.pending_message = None
        reply = call_llm_as_patient(st.session_state.case, st.session_state.history)
        audio_file = tts_mp3(reply)
        st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_file})
        st.rerun()

    left_icons, input_col = st.columns([0.08, 0.92])

    with left_icons:
        kb = st.button("⌨", key="kb_btn", help="Keyboard", use_container_width=True)
        mic = st.button("🎤", key="mic_btn", help="Mic", use_container_width=True)
        if kb: st.session_state.input_mode = "keyboard"
        if mic: st.session_state.input_mode = "voice"

    with input_col:
        if st.session_state.input_mode == "keyboard":
            user_text = st.chat_input("Type your question…")
            if user_text:
                st.session_state.history.append({"role": "user", "content": user_text})
                st.session_state.pending_message = user_text   # 🟡 store message for next run
                st.rerun()
        else:
            audio_data = st.audio_input("Record your question", label_visibility="collapsed")
            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data.read())
                    f.flush()
                    path = f.name
                transcript = speech_to_text(path)
                st.session_state.history.append({"role": "user", "content": transcript, "audio": path})
                st.session_state.pending_message = transcript
                st.rerun()

    def handle_patient_reply():
        try:
            reply = call_llm_as_patient(st.session_state.case, st.session_state.history)
            audio_file = tts_mp3(reply)
            st.session_state.history.append({"role": "assistant", "content": reply, "audio": audio_file})
        except Exception as e:
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        st.session_state.is_replying = False
        st.rerun()

    left_icons, input_col = st.columns([0.08, 0.92])

    with left_icons:
        kb = st.button("⌨", key="kb_btn", help="Keyboard", use_container_width=True)
        mic = st.button("🎤", key="mic_btn", help="Mic", use_container_width=True)
        if kb: st.session_state.input_mode = "keyboard"
        if mic: st.session_state.input_mode = "voice"

    with input_col:
        if st.session_state.input_mode == "keyboard":
            user_text = st.chat_input("Type your question…")
            if user_text:
                st.session_state.history.append({"role": "user", "content": user_text})
                if not st.session_state.is_replying:
                    st.session_state.is_replying = True
                    threading.Thread(target=handle_patient_reply, daemon=True).start()
                st.rerun()

        else:
            audio_data = st.audio_input("Record your question", label_visibility="collapsed")
            if audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(audio_data.read())
                    f.flush()
                    path = f.name
                transcript = speech_to_text(path)
                st.session_state.history.append({"role": "user", "content": transcript, "audio": path})
                if not st.session_state.is_replying:
                    st.session_state.is_replying = True
                    threading.Thread(target=handle_patient_reply, daemon=True).start()
                st.rerun()

    def build_transcript_file():
        buf = io.StringIO()
        buf.write(f"Sultan Qaboos University – Clinical Skills Lab\n")
        buf.write(f"Encounter Transcript\nPatient: {st.session_state.patient_name}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        for msg in st.session_state.history:
            speaker = "Doctor" if msg["role"] == "user" else "Patient"
            buf.write(f"{speaker}: {msg['content']}\n")
        return buf.getvalue().encode("utf-8")

    st.download_button(
        label="💾 End Encounter & Download",
        data=build_transcript_file(),
        file_name=f"encounter_{st.session_state.patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )
