import streamlit as st
from PyPDF2 import PdfReader

from agents import (
    profile_analysis_node,
    job_fit_node,
    content_enhancement_node,
    career_guide_node
)

st.set_page_config(page_title="LinkedIn Career Coach", layout="wide")
st.title("🤖 LinkedIn Career Coach Assistant")

uploaded_file = st.file_uploader("📤 Upload your LinkedIn Profile PDF", type=["pdf"])
job_title = st.text_input("🎯 Target Job Title")
user_question = st.text_area("💬 Ask a Career Question (for Career Guide tab)")

def extract_profile_sections_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    sections = {
        "about": "", "experience": "", "skills": "", "education": "", "projects": "",
        "certifications": "", "publications": "", "volunteering": "", "awards": "",
        "languages": "", "courses": ""
    }

    current_section = None
    for line in full_text.splitlines():
        line_clean = line.strip().lower()
        for section in sections:
            if section in line_clean:
                current_section = section
                break
        else:
            if current_section:
                sections[current_section] += line_clean + " "
    return sections

if uploaded_file:
    profile = extract_profile_sections_from_pdf(uploaded_file)
    st.success("✅ Profile uploaded and parsed!")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Profile Analysis", "🎯 Job Fit", "✨ Content Enhancement", "🧭 Career Guide"])

    # --------------------------- TAB 1: PROFILE ANALYSIS ---------------------------
    with tab1:
        if st.button("Run Profile Analysis"):
            with st.spinner("Running Gemini analysis..."):
                try:
                    analysis_result = profile_analysis_node.invoke({"profile": profile})
                    st.subheader("📊 Profile Review")
                    st.markdown(analysis_result["analysis"])
                except Exception as e:
                    st.error(f"❌ Failed to analyze: {e}")

    # --------------------------- TAB 2: JOB FIT ANALYSIS ---------------------------
    with tab2:
        if st.button("Run Job Fit Analysis"):
            with st.spinner("Matching your profile with live job listings..."):
                try:
                    job_fit_result = job_fit_node.invoke({"profile": profile, "job_title": job_title})
                    st.subheader("🎯 Job Fit Analysis")
                    st.markdown(job_fit_result["job_fit_result"])

                    if job_fit_result["job_matches"]:
                        st.markdown("#### 💼 Top Matching Jobs")
                        for job in job_fit_result["job_matches"][:3]:
                            st.markdown(f"- {job}")
                except Exception as e:
                    st.error(f"❌ Job Fit failed: {e}")

    # --------------------------- TAB 3: CONTENT ENHANCEMENT ---------------------------
    with tab3:
        if st.button("Enhance Profile Content"):
            with st.spinner("Rewriting profile for better industry alignment..."):
                try:
                    # Must run analysis first
                    analysis_result = profile_analysis_node.invoke({"profile": profile})

                    enhancement_result = content_enhancement_node.invoke({
                        "profile": profile,
                        "job_title": job_title,
                        "analysis": analysis_result["analysis"]
                    })
                    st.subheader("✨ Enhanced Profile Content")
                    st.markdown(enhancement_result["enhanced_content"])

                    st.download_button("📥 Download Enhanced Profile",
                        enhancement_result["enhanced_content"],
                        file_name="enhanced_profile.txt")
                except Exception as e:
                    st.error(f"❌ Content Enhancement failed: {e}")

    # --------------------------- TAB 4: CAREER GUIDE ---------------------------
    with tab4:
        if st.button("Get Career Guidance"):
            with st.spinner("Thinking deeply about your question..."):
                try:
                    analysis_result = profile_analysis_node.invoke({"profile": profile})
                    job_fit_result = job_fit_node.invoke({"profile": profile, "job_title": job_title})
                    enhancement_result = content_enhancement_node.invoke({
                        "profile": profile,
                        "job_title": job_title,
                        "analysis": analysis_result["analysis"]
                    })

                    guide_response = career_guide_node.invoke({
                        "profile": profile,
                        "job_title": job_title,
                        "analysis": analysis_result["analysis"],
                        "job_matches": job_fit_result["job_matches"],
                        "job_fit_result": job_fit_result["job_fit_result"],
                        "enhanced_content": enhancement_result["enhanced_content"],
                        "user_question": user_question
                    })

                    st.subheader("🧭 Career Advice")
                    st.markdown(guide_response["career_guide_response"])
                except Exception as e:
                    st.error(f"❌ Career Guide failed: {e}")
