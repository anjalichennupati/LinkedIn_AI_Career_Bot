import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
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

# State for career plan review
if "career_plan_run" not in st.session_state:
    st.session_state.career_plan_run = None
if "career_plan_draft" not in st.session_state:
    st.session_state.career_plan_draft = None
if "career_plan_final" not in st.session_state:
    st.session_state.career_plan_final = None
if "available_sessions" not in st.session_state:
    st.session_state.available_sessions = []

# Session Management Section
st.markdown("---")
st.subheader("💾 Session Management")

# Load available sessions
try:
    from agents import memory

    if hasattr(memory, "list_keys"):
        available_sessions = memory.list_keys()
        st.session_state.available_sessions = available_sessions
    else:
        st.info("ℹ️ Session persistence not available")
except Exception as e:
    st.info(f"ℹ️ Session loading not available: {e}")

# Show available sessions
if st.session_state.available_sessions:
    st.write("**Available Sessions:**")
    for session_id in st.session_state.available_sessions[:5]:  # Show first 5
        if st.button(
            f"📂 Load Session: {session_id[:20]}...", key=f"load_{session_id}"
        ):
            try:
                # Load the session
                loaded_state = memory.get({"configurable": {"thread_id": session_id}})
                if loaded_state:
                    st.session_state.career_plan_run = loaded_state
                    st.session_state.thread_id = session_id
                    st.success(f"✅ Session {session_id[:20]}... loaded!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load session")
            except Exception as e:
                st.error(f"❌ Error loading session: {e}")
else:
    st.info("ℹ️ No saved sessions found")

linkedin_url = st.text_input("🔗 Enter LinkedIn Profile URL")

from agents import graph  # import the compiled graph


# if st.button("🔍 Scrape LinkedIn"):
#     if not linkedin_url:
#         st.warning("Please enter a LinkedIn profile URL.")
#     else:
#         with st.spinner("Scraping LinkedIn..."):
#             try:
#                 scraped = scrape_and_clean_profile(
#                     linkedin_url, api_token=os.getenv("APIFY_API_KEY")
#                 )
#                 if not scraped:
#                     st.error("❌ Scraping failed or profile was private.")
#                 else:
#                     st.session_state.profile_data = scraped
#                     st.success("✅ LinkedIn profile scraped and parsed!")
#             except Exception as e:
#                 st.error(f"Scraping error: {e}")

# linkedin_url = st.text_input("🔗 Enter LinkedIn Profile URL")

if st.button("🚀 Start Assistant"):
    if not linkedin_url:
        st.warning("Please enter a LinkedIn profile URL.")
    else:
        st.session_state.linkedin_url = linkedin_url

        with st.spinner("Running assistant..."):
            try:
                result = graph.invoke(
                    {"linkedin_url": linkedin_url},
                    config={
                        "configurable": {"thread_id": "default-thread"}
                    },  # 👈 add this
                )
                st.success(result["profile_data"])
                st.session_state.profile_data = result.get("profile_data", {})
                st.success("✅ LinkedIn profile scraped and assistant ready!")
            except Exception as e:
                st.error(f"Error: {e}")


job_description = st.text_area("📝 Paste the job description (optional)", height=200)
user_question = st.text_input(
    "💬 Ask your AI Career Guide anything (Type quit to stop execution)"
)

if st.button("🚀 Ask AI"):
    if not st.session_state.profile_data:
        st.warning("Please scrape a LinkedIn profile first.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                state = {
                    "messages": [HumanMessage(content=user_question)],
                    "profile_data": st.session_state.profile_data,
                    "current_job_description": job_description,
                }

                output = graph.invoke(
                    state,
                    config={
                        "thread_id": st.session_state.thread_id,
                        "checkpoint": memory,
                    },
                )

                ai_messages = [
                    msg
                    for msg in output.get("messages", [])
                    if isinstance(msg, AIMessage)
                ]
                answer = ai_messages[-1].content if ai_messages else "No response."
                st.session_state.chat_history.append((user_question, answer))

            except Exception as e:
                st.error(f"❌ Error: {e}")

if st.button("🧹 Clear Conversation"):
    st.session_state.chat_history = []
    st.session_state.profile_data = None
    st.session_state.linkedin_url = linkedin_url
    st.session_state.thread_id = "career-session"
    st.success("Chat cleared!")

if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Conversation History")
    for question, response in reversed(st.session_state.chat_history):
        st.markdown(f"🧑 **You:** {question}")
        st.markdown(f"🤖 **AI:** {response}")

st.markdown("---")
st.subheader("📝 Career Plan Generator")

# Career plan input section
career_brief = st.text_input(
    "Career Goal Brief",
    placeholder="e.g., Data analyst (Python/SQL) targeting Data Scientist in 6 months",
)

if st.button("🎯 Generate Career Plan"):
    if not career_brief:
        st.warning("Please enter a career goal brief.")
    elif not st.session_state.profile_data:
        st.warning("Please scrape a LinkedIn profile first.")
    else:
        with st.spinner("Generating your career plan..."):
            try:
                # 👇 force include latest scraped profile_data
                state = {
                    "messages": [HumanMessage(content=career_brief)],
                    "profile_data": dict(st.session_state.profile_data),  # force copy
                    "current_job_description": job_description,
                    "linkedin_url": linkedin_url,
                }

                print("DEBUG: state passed to graph:", state)  # 👀 verify

                output = graph.invoke(
                    state,
                    config={
                        "thread_id": st.session_state.thread_id,
                        "checkpoint": memory,
                    },
                )

                st.success(output)

                st.session_state.career_plan_run = output
                # Extract the plan from messages
                messages = output.get("messages", [])
                ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                if ai_msgs:
                    st.session_state.career_plan_draft = ai_msgs[-1].content
                    st.success("✅ Career plan generated! Review and refine below.")
                    st.success(st.session_state.career_plan_draft)
                else:
                    st.error("❌ No plan generated.")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# Career plan review section
if st.session_state.career_plan_draft:
    st.markdown("---")
    st.subheader("📋 Review & Refine Your Plan")

    st.markdown("**Generated Plan:**")
    st.markdown(st.session_state.career_plan_draft)

    review_notes = st.text_area(
        "Suggestions for improvement (optional)",
        placeholder="e.g., Make it 3 months only, add more networking tasks, focus on ML projects",
    )

    if st.button("✅ Apply Feedback"):
        if not st.session_state.career_plan_run:
            st.warning("No career plan to refine.")
        else:
            with st.spinner("Applying your feedback..."):
                try:
                    feedback_notes = review_notes.strip()
                    if not feedback_notes:
                        st.warning("Please enter some feedback to apply.")
                    else:
                        # Prepare state for LangGraph to process feedback
                        state = {
                            "messages": [HumanMessage(content=feedback_notes)],
                            "profile_data": st.session_state.profile_data,
                            "current_job_description": job_description,
                            "processing_feedback": True,  # Flag for career_plan_node
                        }

                        # Invoke the graph so the backend handles feedback & persistence
                        output = graph.invoke(
                            state,
                            config={
                                "thread_id": st.session_state.thread_id,
                                "checkpoint": memory,
                            },
                        )

                        # Extract the refined plan from AI messages
                        ai_msgs = [
                            msg
                            for msg in output.get("messages", [])
                            if isinstance(msg, AIMessage)
                        ]
                        if ai_msgs:
                            st.session_state.career_plan_final = ai_msgs[-1].content
                            st.success("✅ Plan refined with your feedback!")
                        else:
                            st.error("❌ No refined plan generated. Check logs.")
                except Exception as e:
                    st.error(f"❌ Error applying feedback: {e}")

    # Show the final plan
    if st.session_state.career_plan_final:
        st.subheader("🎯 Final Refined Plan")
        st.markdown(st.session_state.career_plan_final)

        # Add Request Review button
        if st.button("🔄 Request Review", type="secondary"):
            # Reset to allow new review
            st.session_state.career_plan_final = None
            st.session_state.career_plan_draft = None
            st.success(
                "✅ Ready for new review! Generate a new plan or modify existing one."
            )
            st.rerun()
