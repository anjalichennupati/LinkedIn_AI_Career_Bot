# agents.py

import google.generativeai as genai
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda

genai.configure(api_key="AIzaSyDX9htz1H9osUdf-RTN-z4DfiMLnbSUPkQ")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# DataState
class ProfileState(TypedDict):
    profile: Annotated[dict, lambda _, x: x]
    job_description: Annotated[str, lambda _, x: x]
    user_question: Annotated[str, lambda _, x: x]
    analysis: Annotated[str, lambda _, x: x]
    job_fit_feedback: Annotated[str, lambda _, x: x]
    enhancement: Annotated[str, lambda _, x: x]
    answer: Annotated[str, lambda _, x: x]

def analyze_profile(state: ProfileState) -> ProfileState:
    profile = state["profile"]
    prompt = "Analyze this LinkedIn profile:\n"
    for sec, val in profile.items():
        prompt += f"## {sec.capitalize()}:\n{val.strip()}\n"
    prompt += "\nGive strengths, weaknesses, and 3 improvement tips."
    result = model.generate_content(prompt)
    return {**state, "analysis": result.text}

def job_fit_agent(state: ProfileState) -> ProfileState:
    if not state.get("job_description"):
        return state

    profile_text = "\n".join(f"## {sec.capitalize()}:\n{val}" for sec, val in state["profile"].items())
    prompt = f"""
You are a Job Fit Analysis AI.

--- JOB DESCRIPTION ---
{state["job_description"]}

--- CANDIDATE PROFILE ---
{profile_text}

Instructions:
- Give a *Job Match Score out of 100*
- List 3–5 key strengths
- Suggest 3–5 improvements
"""
    response = model.generate_content(prompt)
    return {**state, "job_fit_feedback": response.text}

def enhance_profile(state: ProfileState) -> ProfileState:
    profile_text = "\n".join(f"## {sec.capitalize()}:\n{val}" for sec, val in state["profile"].items())

    jd = state.get("job_description", "")
    jd_context = f"to better align with this job description:\n{jd}" if jd.strip() else "to generally improve it."

    prompt = f"""
You are a LinkedIn Profile Enhancement AI.
Enhance this profile {jd_context}

--- PROFILE ---
{profile_text}
"""
    result = model.generate_content(prompt)
    return {**state, "enhancement": result.text}

def career_qa_agent(state: ProfileState) -> ProfileState:
    question = state["user_question"].lower()

    if any(x in question for x in ["gap", "improve", "weak", "review", "career path", "strength"]):
        state = analyze_profile(state)

    if any(x in question for x in ["fit", "score", "description", "match", "suit"]):
        state = job_fit_agent(state)

    if any(x in question for x in ["enhance", "rewrite", "optimize", "improve my profile"]):
        state = enhance_profile(state)

    prompt = f"""
You are a professional AI Career Coach.
User's Question: {state['user_question']}

--- PROFILE SUMMARY ---
{state['profile'].get('about', '')[:1000]}

--- ANALYSIS ---
{state.get('analysis', '')}

--- JOB FIT ---
{state.get('job_fit_feedback', '')}

--- ENHANCED VERSION ---
{state.get('enhancement', '')}

Respond helpfully, practically, and in a personalized tone.
"""
    response = model.generate_content(prompt)
    return {**state, "answer": response.text}

def route(state: ProfileState):
    return "qa"

# LangGraph
memory = MemorySaver()
builder = StateGraph(ProfileState)

builder.add_node("qa", RunnableLambda(career_qa_agent))

builder.set_entry_point("qa")
builder.set_finish_point("qa")

graph = builder.compile()
