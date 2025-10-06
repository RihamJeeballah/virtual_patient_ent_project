import streamlit as st
import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv

# ======== LOAD API KEY ========
load_dotenv()  # reads .env file in the same folder
client = OpenAI()  # automatically uses OPENAI_API_KEY

# ======== CONFIG ========
CASES_DIR = Path("cases")
LOGS_DIR = Path("conversations")
LOGS_DIR.mkdir(exist_ok=True)
MODEL = "gpt-4o-mini"  # or change to "gpt-4o" if available


# ======== HELPERS ========
def load_case(file_path: Path) -> Dict[str, str]:
    """Parse markdown sections into a dictionary."""
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
    """Call LLM API using conversation history (OpenAI SDK v1+)."""
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
        return "⚠️ Sorry, I couldn’t generate a response. Please check your OpenAI key or model name."


def summarize_chat(history):
    """Simple encounter summary."""
    text = "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])
    return f"# Encounter Summary\n\n{text}"


# ======== STREAMLIT UI ========
st.set_page_config(page_title="Virtual ENT Patient", layout="centered")

# --- Header ---
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
st.title("🩺 Virtual ENT Patient")

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

    # ======== Auto-start patient ========
    if not st.session_state.started:
        greeting_prompt = f"""
You are the patient in this ENT clinical case.
Greet the doctor naturally and briefly describe your main complaint
based only on the case details below. Speak as the patient.

Case details:
{json.dumps(case, indent=2)}
"""
        greeting = call_llm(greeting_prompt, [])
        st.session_state.history.append({"role": "assistant", "content": greeting})
        st.session_state.started = True

    # ======== Display chat bubbles ========
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ======== Doctor input ========
    if prompt := st.chat_input("Ask your patient..."):
        st.session_state.history.append({"role": "user", "content": prompt})

        system_prompt = f"""
You are simulating the patient described in the following clinical case.
Only use information contained in the case file.
Respond naturally as the patient, revealing details only when asked.

Case details:
{json.dumps(case, indent=2)}
"""
        reply = call_llm(system_prompt, st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)

    # ======== End encounter ========
    if st.button("End Encounter"):
        summary = summarize_chat(st.session_state.history)
        st.session_state.summary = summary

        # Save conversation
        log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{st.session_state.case_name}.json"
        with open(LOGS_DIR / log_name, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, indent=2)

        st.markdown("### 📝 Encounter Summary")
        st.markdown(summary)
        st.download_button("Download Summary (.md)", summary, file_name="summary.md")

        st.session_state.case = None
