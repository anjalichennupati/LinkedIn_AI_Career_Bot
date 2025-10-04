import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agents import qa_graph, memory
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

st.set_page_config(page_title="AI Career Chat Assistant", layout="wide")
st.title("ðŸ’¼ Chat-based AI Career Assistant")

# Initialize session state with proper defaults
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "profile_data" not in st.session_state:
    st.session_state.profile_data = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"career-session-{uuid.uuid4().hex[:8]}"

if "linkedin_url" not in st.session_state:
    st.session_state.linkedin_url = ""

if "profile_scraped" not in st.session_state:
    st.session_state.profile_scraped = False

# Career plan review state
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
st.subheader("ðŸ—‚ Session Management")

try:
    if hasattr(memory, "list_keys"):
        available_sessions = memory.list_keys()
        st.session_state.available_sessions = available_sessions
    else:
        st.info("â„¹ Session persistence not available")
except Exception as e:
    st.info(f"â„¹ Session loading not available: {e}")

# Show available sessions
if st.session_state.available_sessions:
    st.write("Available Sessions:")
    for session_id in st.session_state.available_sessions[:5]:
        if st.button(f"ðŸ“„ Load Session: {session_id[:20]}...", key=f"load_{session_id}"):
            try:
                loaded_state = memory.get({"configurable": {"thread_id": session_id}})
                if loaded_state:
                    st.session_state.career_plan_run = loaded_state
                    st.session_state.thread_id = session_id
                    st.success(f"âœ… Session {session_id[:20]}... loaded!")
                    st.rerun()
                else:
                    st.error("âŒ Failed to load session")
            except Exception as e:
                st.error(f"âŒ Error loading session: {e}")
else:
    st.info("â„¹ No saved sessions found")

# LinkedIn URL Input
linkedin_url = st.text_input(
    "ðŸ“Š Enter LinkedIn Profile URL", 
    value=st.session_state.get("linkedin_url", ""),
    key="linkedin_input"
)

# FIX 1: Only scrape if URL changed or profile not yet scraped
if st.button("ðŸš€ Start Assistant"):
    if not linkedin_url:
        st.warning("Please enter a LinkedIn profile URL.")
    else:
        # Check if we need to scrape (URL changed or not scraped yet)
        need_to_scrape = (
            not st.session_state.profile_scraped or 
            linkedin_url != st.session_state.linkedin_url or 
            not st.session_state.profile_data
        )
        
        if need_to_scrape:
            # Generate new thread ID only when starting fresh
            st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
            st.session_state.linkedin_url = linkedin_url

            with st.spinner("Running assistant..."):
                try:
                    st.image(qa_graph.get_graph().draw_mermaid_png())
                    result = qa_graph.invoke(
                        {
                            "linkedin_url": linkedin_url,
                            "thread_id": st.session_state.thread_id,
                            "messages": [],
                            "profile_scraped": False  # Trigger scraping
                        },
                        config={
                            "configurable": {"thread_id": st.session_state.thread_id}
                        },
                    )
                    
                    st.session_state.profile_data = result.get("profile_data", {})
                    st.session_state.profile_scraped = True
                    st.success("âœ… LinkedIn profile scraped and assistant ready!")
                    st.info(f"Session ID: {st.session_state.thread_id}")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.error(traceback.format_exc())
        else:
            st.info("âœ… Profile already loaded! Ready to chat.")
            st.info(f"Current Session ID: {st.session_state.thread_id}")

# Show profile status
if st.session_state.profile_data:
    profile_headline = st.session_state.profile_data.get("headline", "Unknown")
    st.success(f"ðŸ‘¤ Profile loaded: {profile_headline}")

# Job Description and User Question
job_description = st.text_area("Paste the job description (optional)", height=200)
user_question = st.text_input(
    "Ask your AI Career Guide anything (Type quit to stop execution)"
)

# FIX 1: Improved Ask AI - no re-scraping, direct to router
if st.button("ðŸ” Ask AI"):
    if not st.session_state.profile_data:
        st.warning("Please scrape a LinkedIn profile first.")
    elif not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                # FIX 1: Start from router, not scraper, since profile already exists
                state = {
                    "messages": [HumanMessage(content=user_question)],
                    "profile_data": {**st.session_state.profile_data, "_graph_type": "qa_graph"},

                    "current_job_description": job_description,
                    "thread_id": st.session_state.thread_id,
                    "profile_scraped": True  # Profile already scraped
                }

                # FIX 1: Invoke starting from career_qa_router to avoid re-scraping
                from langgraph.types import Command
                
                # Create a minimal graph state and invoke the router directly
                output = qa_graph.invoke(
                    state,
                    config={
                        "configurable": {"thread_id": st.session_state.thread_id}
                    }
                )

                ai_messages = [
                    msg for msg in output.get("messages", [])
                    if isinstance(msg, AIMessage)
                ]
                answer = ai_messages[-1].content if ai_messages else "No response."
                st.session_state.chat_history.append((user_question, answer))

            except Exception as e:
                st.error(f"âŒ Error: {e}")
                import traceback
                st.error(traceback.format_exc())

# Clear conversation button
if st.button("ðŸ—‘ Clear Conversation"):
    st.session_state.chat_history = []
    # Don't clear profile data, just conversation
    st.success("Chat cleared!")

# Display conversation history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("ðŸ—£ Conversation History")
    for question, response in reversed(st.session_state.chat_history):
        st.markdown(f"ðŸ¤” You: {question}")
        st.markdown(f"ðŸ’¬ AI: {response}")

st.markdown("---")
st.subheader("🤖 Unified Assistant (Chat + Planning)")
st.caption("All interactions go through a single graph with interrupt/resume.")

st.markdown("---")
st.subheader("ðŸ”§ Procedural Learning Debug")

# if st.button("ðŸ” Debug Current Thread"):
#     if hasattr(st.session_state, 'thread_id'):
#         try:
#             thread_procedural_memory.debug_thread(st.session_state.thread_id)
#             st.success(f"Check console for debug info of thread: {st.session_state.thread_id}")
#         except Exception as e:
#             st.error(f"Debug failed: {e}")
#     else:
#         st.warning("No active thread found")

# if st.button("ðŸ§ª Test Procedural Learning"):
#     try:
#         test_thread_based_learning()
#         st.success("Test completed - check console for results")
#     except Exception as e:
#         st.error(f"Test failed: {e}")

# # Add memory statistics display
# if st.button("ðŸ“Š Show Thread Statistics"):
#     if hasattr(st.session_state, 'thread_id') and st.session_state.profile_data:
#         try:
#             user_id = st.session_state.profile_data.get("user_id", "unknown")
#             if hasattr(thread_procedural_memory, 'get_user_stats'):
#                 stats = thread_procedural_memory.get_user_stats(user_id)
#                 st.json(stats)
#             else:
#                 st.warning("User stats method not available")
#         except Exception as e:
#             st.error(f"Stats retrieval failed: {e}")

# # Complete reset button
# if st.button("ðŸ”„ Complete Reset"):
#     st.session_state.chat_history = []
#     st.session_state.profile_data = None
#     st.session_state.profile_scraped = False
#     st.session_state.linkedin_url = ""
#     st.session_state.career_plan_draft = None
#     st.session_state.career_plan_final = None
#     st.session_state.career_plan_run = None
#     # Generate new thread ID for fresh start
#     st.session_state.thread_id = f"career-session-{uuid.uuid4().hex[:8]}"
#     st.success("Complete reset done! Ready for new session.")
#     st.rerun()