# app.py

import streamlit as st
from PyPDF2 import PdfReader
from agents import graph, memory

st.set_page_config(page_title="AI Career Chat Assistant", layout="wide")
st.title("💼 Chat-based AI Career Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "profile_data" not in st.session_state:
    st.session_state.profile_data = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "career-session"

uploaded_file = st.file_uploader("📤 Upload your LinkedIn Profile (PDF)", type=["pdf"])

def extract_profile_sections(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    sections = {
        "about": "", "experience": "", "skills": "",
        "education": "", "projects": "", "certifications": "",
        "publications": "", "volunteering": "", "awards": "",
        "languages": "", "courses": ""
    }

    current_section = None
    for line in full_text.splitlines():
        line_clean = line.strip()
        line_lower = line_clean.lower()
        for section in sections.keys():
            if section in line_lower:
                current_section = section
                break
        else:
            if current_section:
                sections[current_section] += line_clean + " "

    return sections

if uploaded_file:
    st.session_state.profile_data = extract_profile_sections(uploaded_file)
    st.success("✅ Profile parsed successfully!")

job_description = st.text_area("📝 Paste the job description (optional)")
user_question = st.text_input("💬 Ask your AI Career Guide anything")

if st.button("🚀 Ask AI"):
    if not st.session_state.profile_data:
        st.warning("Please upload a LinkedIn profile first.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                output = graph.invoke(
                    {
                        "profile": st.session_state.profile_data,
                        "job_description": job_description,
                        "user_question": user_question
                    },
                    config={
                        "thread_id": st.session_state.thread_id,
                        "checkpoint": memory
                    }
                )

                answer = output.get("answer", "No response.")
                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                st.error(f"❌ Error: {e}")

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Conversation History")
    for question, response in reversed(st.session_state.chat_history):
        st.markdown(f"🧑 You:** {question}")
        st.markdown(f"🤖 AI:** {response}")
