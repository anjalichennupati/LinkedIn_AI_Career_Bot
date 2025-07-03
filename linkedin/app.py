# app.py

import streamlit as st
from agents import graph, memory
from scraper_utils import scrape_and_clean_profile

import os
from dotenv import load_dotenv

load_dotenv() 

st.set_page_config(page_title="AI Career Chat Assistant", layout="wide")
st.title("💼 Chat-based AI Career Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "profile_data" not in st.session_state:
    st.session_state.profile_data = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "career-session"

linkedin_url = st.text_input("🔗 Enter LinkedIn Profile URL")

if st.button("🔍 Scrape LinkedIn"):
    if not linkedin_url:
        st.warning("Please enter a LinkedIn profile URL.")
    else:
        with st.spinner("Scraping LinkedIn..."):
            try:
                scraped = scrape_and_clean_profile(
                    linkedin_url, 
                    api_token=os.getenv("APIFY_API_KEY")
                )
                if not scraped:
                    st.error("❌ Scraping failed or profile was private.")
                else:
                    st.session_state.profile_data = scraped
                    st.success("✅ LinkedIn profile scraped and parsed!")
            except Exception as e:
                st.error(f"Scraping error: {e}")

job_description = st.text_area("📝 Paste the job description (optional)")
user_question = st.text_input("💬 Ask your AI Career Guide anything")

if st.button("🚀 Ask AI"):
    if not st.session_state.profile_data:
        st.warning("Please scrape a profile first.")
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
