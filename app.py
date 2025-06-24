# app.py
import streamlit as st
from linkedin_scraper import scrape_linkedin_profile
from agents.profile_analyzer import analyze_profile
from agents.memory_manager import load_memory, update_memory

st.set_page_config(page_title="AI LinkedIn Coach")
st.title("\U0001F4BB AI Career Bot for LinkedIn Optimization")

# Initialize memory (dict based)
memory = load_memory()

# Step 1: LinkedIn URL Input
profile_url = st.text_input("Paste your LinkedIn Profile URL:")

if profile_url:
    with st.spinner("Scraping LinkedIn profile..."):
        profile_data = scrape_linkedin_profile(profile_url)
        memory['profile_data'] = profile_data
        update_memory(memory)
        st.success("Profile scraped successfully!")

# Step 2: Display Profile Summary
if 'profile_data' in memory:
    st.subheader("Extracted Profile Summary:")
    st.json(memory['profile_data'])

    # Step 3: Ask Job Role
    job_role = st.text_input("Target Job Role (e.g. Data Scientist):")
    if job_role:
        with st.spinner("Analyzing profile for job fit..."):
            feedback, match_score = analyze_profile(memory['profile_data'], job_role)
            memory['feedback'] = feedback
            memory['match_score'] = match_score
            update_memory(memory)

        st.subheader("\U0001F4CB Profile Feedback:")
        st.markdown(feedback)
        st.metric(label="\U0001F4CA Match Score", value=f"{match_score} / 100")

        # Chat Interface (simplified text box)
        st.subheader("\U0001F5E3 Career Q&A Chat")
        query = st.text_input("Ask a career or profile-related question:")
        if query:
            st.write("\U0001F916 Thinking...")
            from langchain.chat_models import ChatOpenAI
            llm = ChatOpenAI()
            response = llm.predict(f"{query}\nHere's my profile: {memory['profile_data']}\n{feedback}")
            st.markdown(response)
