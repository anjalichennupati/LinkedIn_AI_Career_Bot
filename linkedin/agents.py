import os
import json
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai
from scraper_utils import scrape_and_clean_profile

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    profile_data: Annotated[dict, lambda _, x: x]
    current_job_description: Annotated[Optional[str], lambda _, x: x]


def career_qa_router(state: AgentState) -> Command:
    profile = state.get("profile_data", {})
    jd = state.get("current_job_description", "")
    messages = state.get("messages", [])
    question = ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content.strip()
            break

    if question.lower() in {"quit", "exit", "stop"}:
        messages.append(AIMessage("Okay, ending the conversation."))
        return Command(goto=END, update={"messages": messages})

    if "job description:" in question.lower():
        messages.append(AIMessage("Got your new job description."))
        state["messages"] = messages
        state["current_job_description"] = question
        return interrupt("continue_chat")

    if "linkedin.com/in/" in question.lower():
        messages.append(AIMessage("Got it! Scraping your profile now..."))
        try:
            scraped = scrape_and_clean_profile(
                linkedin_url=question, api_token=os.getenv("APIFY_API_TOKEN")
            )
            if not scraped:
                messages.append(AIMessage("⚠️ Failed to extract profile."))
                state["messages"] = messages
                return interrupt("continue_chat")
            messages.append(AIMessage("✅ Profile successfully scraped!"))
            state["messages"] = messages
            state["profile_data"] = scraped
            return interrupt("continue_chat")
        except Exception as e:
            messages.append(AIMessage(f"❌ Error scraping LinkedIn: {e}."))
            state["messages"] = messages
            return interrupt("continue_chat")

    history = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages[-5:]
    )
    prompt = f"""
    {history}
You are a routing agent. Decide which module should handle the user's latest question.

Return one of:
- analyze_profile
- job_fit_agent
- enhance_profile
- general_qa

DO NOT GUESS. Use the following rules:

---

ROUTE TO: **analyze_profile**
→ If the user wants a LinkedIn/resume/profile review, feedback, strengths, weaknesses, or audit.

Examples:
- “Can you review my LinkedIn?”
- “What are my strengths and weaknesses?”
- “Audit my profile”
- Or any other question that implies analyzing the profile.
---

ROUTE TO: **job_fit_agent**
→ If the user says anything like:
- “Does my profile match this JD?”
- “Am I eligible for this job?”
- “Score me against this role”
- Or any other question that implies matching the profile to a job description.

*Only* route here if a job description was recently provided.

---

ROUTE TO: **enhance_profile**
→ If the user asks for:
- Rewriting/resume improvement
- Profile optimization
- “Improve my About section”
- “Rewrite my Experience bullets”
- Or any other question that implies enhancing the profile.

---

ROUTE TO: **general_qa**
→ All other general career questions.

Examples:
- “What kind of roles should I target?”
- “How do I switch fields?”
- “What are good certifications for data science?”
- “How do I get into startups?”
- Or any other question that doesn't fit the above categories.

---

USER QUESTION:
{question}

Just respond with ONE of:
analyze_profile, job_fit_agent, enhance_profile, general_qa
"""

    try:
        result = model.generate_content(prompt)
        decision = result.text.strip().lower()

        if decision in {
            "analyze_profile",
            "job_fit_agent",
            "enhance_profile",
            "general_qa",
        }:
            state["messages"] = messages  

            return Command(
                goto=decision if decision != "general_qa" else "general_qa_node"
            )

        else:
            messages.append(AIMessage("⚠️ I didn’t understand. Try rephrasing."))
            state["messages"] = messages
            return interrupt("continue_chat")
    except Exception as e:
        messages.append(AIMessage(f"⚠️ Routing error: {e}"))
        state["messages"] = messages
        return interrupt("continue_chat")


# 🔍 analyze_profile_node
def analyze_profile_node(state: AgentState) -> dict:
    profile = state.get("profile_data")
    messages = state["messages"]
    if not profile:
        messages.append(AIMessage("⚠️ No profile data found to analyze."))
        state["messages"] = messages
        return interrupt("continue_chat")

    profile_text = "\n".join(f"{k}: {v}" for k, v in profile.items())
    history = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages[-5:]
    )

    prompt = f"""
{history}
        You are a highly experienced career coach and tech recruiter who has reviewed over 10,000 LinkedIn profiles.

        Your job is to critically evaluate the following LinkedIn profile and return a brutally honest, section-wise analysis.

        Use the following criteria for evaluation:
        1. Clarity and professionalism of writing
        2. Technical and strategic relevance of content
        3. Recruiter impression: Would you shortlist this profile?

        ---

        {profile_text}

        ---

        ### Return output in the following structure:

        # LinkedIn Profile Audit

        ## Strengths  
        List 3–5 strengths that stand out across the profile.

        ## Weaknesses  
        List 3–5 major weaknesses holding the profile back.
 ## Section-by-Section Evaluation  
        For each section (About, Experience, Projects, Education, Skills, etc.), write:

        - Give a quality score: ✅ Strong / ⚠ Needs improvement / ❌ Missing  
        - Provide 2–3 suggestions to improve the section (content, phrasing, structure)  
        - Use clean formatting: bold headings, bullet points, and avoid unnecessary repetition.

        Constraints:
        - Max *4 bullet points* per section  
        - Each bullet: *<30 words*  
        - Total section feedback: *<100 words*

        ## Top 3 Improvements You Must Make Now  
        Each point must be:
        - Brutally specific  
        - Directly actionable  
        - One line only
        
        Precautions:
        - Do not hallucinate and stay within context.
        - If the user asks for/about a specific section (e.g., "improve my projects"), focus only on that section.
        
        Begin now.
        """.strip()

    try:
        res = model.generate_content(prompt)
        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"❌ Error: {e}"))
        state["messages"] = messages

    return interrupt("continue_chat")


def job_fit_agent_node(state: AgentState) -> dict:
    profile = state.get("profile_data")
    jd = state.get("current_job_description", "")
    messages = state["messages"]
    if not profile or not jd:
        messages.append(AIMessage("⚠️ Missing profile or job description."))
        state["messages"] = messages
        return interrupt("continue_chat")

    profile_text = "\n".join(f"{k}: {v}" for k, v in profile.items())
    history = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages[-5:]
    )

    prompt = f"""
{history}
        You are a highly experienced AI Job Fit Evaluator trained on thousands of hiring decisions across several job roles.
        Your job is to evaluate how well the candidate's profile matches the given job description.       
        ---

        JOB DESCRIPTION:
        {jd}

        CANDIDATE PROFILE:
        {profile_text}

        ---

        TASKS:
        1. *Evaluate the fitness of the candidate for this job role* using industry-standard evaluation practices (skills, experience, keywords, impact, achievements, and alignment).
        2. *Return a Job Match Score out of 100*, and explain how you arrived at it with specific reasoning.
        3. *List 3–5 strengths* from the candidate’s profile that match the job expectations.
        4. *Suggest 3–5 concrete improvements* — these could include skill gaps, experience tweaks, weak areas in phrasing, or missing proof of impact.
  5. Only evaluate against the *given job role*. Do not assume adjacent job titles are valid matches.
        6. If the candidate seems overqualified or underqualified, clearly state it and explain how that affects the match.

        ---

        OUTPUT FORMAT:
        # 🎯 Job Fit Evaluation

        ## ✅ Job Match Score: XX/100
        - One-line explanation of the score.
        - 2–3 bullets with specific justification.

        ## 🟩 Strengths
        - Point 1 (aligned with JD)
        - Point 2
        - Point 3  
        (Each point ≤ 40 words)

 ## 🟥 Weaknesses
        - specify the top 3-4 points as to why this profile doesn't match the job or will get rejected even if applied and this analysis must be honest and brutal
        (Each point ≤ 40 words)

        ## 🛠 Improvements to Increase Match Score
        - Point 1 (what to improve and how)
        - Point 2
        - Point 3  
        (Each point ≤ 25 words)

        ## 📌 Verdict
        Clearly say if the candidate is a strong match, weak match, or needs improvement to apply. Give a one-liner summary.
        
        Precautions:
        - Do not hallucinate and stay within context.
        - If the user asks for/about a specific section (e.g., "improve my projects"), focus only on that section.
        
        Begin now.
""".strip()

    try:
        res = model.generate_content(prompt)
        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"❌ Error: {e}"))
        state["messages"] = messages

    return interrupt("continue_chat")


def enhance_profile_node(state: AgentState) -> dict:
    profile = state.get("profile_data")
    jd = state.get("current_job_description", "")
    messages = state["messages"]
    if not profile:
        messages.append(AIMessage("⚠️ No profile found to enhance."))
        state["messages"] = messages
        return interrupt("continue_chat")

    profile_text = "\n".join(f"{k}: {v}" for k, v in profile.items())
    history = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages[-5:]
    )

    prompt = f"""
{history}
        You are a highly experienced LinkedIn Profile Optimization AI, trained on millions of real-world hiring patterns across top companies like Google, Amazon, Meta, and startups. Your task is to analyze and rewrite the user's LinkedIn profile to improve clarity, strength, and impact.

        CONTEXT:
        - The user may or may not have shared a job description.
        - They might be asking to improve a specific section only (e.g., "improve my projects to match this JD").
        - You should infer goals from the user's question and adjust accordingly.
        - If any question is related to the job description/JD then consider that and then provide an output

        CURRENT PROFILE:
        {profile_text}  

        JOB DESCRIPTION/JD:
        {jd}
  TASKS:
        1. Identify weak sections and rewrite them to be stronger, more professional, and better aligned with either:
            - the job description (if provided), or
            - general hiring best practices (if no JD is given).
        2. Preserve all factual details. Do NOT add imaginary experiences.
        3. Use bullet points *only where appropriate* (e.g., Experience, Projects).
        4. Each bullet must be *≤ 25 words, and **max 4 bullets per section*.
        5. For the “About” section, limit to *2–3 tight paragraphs, total **≤ 250 words*.
        6. Add impactful verbs, metrics, and proof of value wherever possible.
        7. If the user requested only a section enhancement (e.g., just projects), *modify only that section*.
        8. If the user asks for a specific section (e.g., "improve my projects"), focus only on that section, DO NOT TALK ABOUT OTHER SECTIONS.

        FORMAT:
        Return the improved sections in clean Markdown format.
        - Use bold section titles.
        - Show only modified sections — skip untouched ones to save tokens.
        - Make sure the rewritten content feels real, focused, and hiring-ready.

Precautions:
        - Do not hallucinate and stay within context.
        - If the user asks for/about a specific section (e.g., "improve my projects"), focus only on that section.

        Begin now.
""".strip()

    try:
        res = model.generate_content(prompt)
        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"❌ Error: {e}"))
        state["messages"] = messages

    return interrupt("continue_chat")


def general_qa_node(state: AgentState) -> dict:
    messages = state["messages"]
    question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            question = msg.content.strip()
            break

    history = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in messages[-5:]
    )

    prompt = f"""
You are a helpful, concise, and highly experienced career guidance assistant.

Your task is to answer **general career-related questions** from users

The user may ask about:
- Career advice
- Interview preparation
- Certifications
- Job search strategy
- Skill-building
- Remote work
- Career switches
- Industry trends
- Anything else loosely related to career growth

---

USER CONTEXT (optional):
{history}

PROFILE DATA:
{state.get("profile_data", "N/A")}

JOB DESCRIPTION (if any):
{state.get("current_job_description", "N/A")}

QUESTION:
{question}

---

### Answer Guidelines:

- Answer **clearly and concisely**.
- Prioritize **useful, actionable advice**.
- If the question is vague or broad, **ask a clarifying follow-up**.
- Keep the tone **supportive but professional**.
- Do **not** suggest uploading a resume or LinkedIn again.
- If you detect the user is **stressed, confused, or unsure**, acknowledge that supportively.

---

### Output Format:

Respond in clean text. Use **bullet points or short paragraphs** where needed.

Start now.
""".strip()

    try:
        res = model.generate_content(prompt)
        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"❌ Error: {e}"))
        state["messages"] = messages

    return interrupt("continue_chat")


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("career_qa_router", career_qa_router)
    builder.add_node("analyze_profile", analyze_profile_node)
    builder.add_node("job_fit_agent", job_fit_agent_node)
    builder.add_node("enhance_profile", enhance_profile_node)
    builder.add_node("general_qa_node", general_qa_node)
    builder.set_entry_point("career_qa_router")

    # Loop back from all task nodes to router
    for task in [
        "analyze_profile",
        "job_fit_agent",
        "enhance_profile",
        "general_qa_node",
    ]:
        builder.add_edge(task, "career_qa_router")

    return builder.compile()


memory = MemorySaver()
graph = build_graph()

if __name__ == "__main__":
    state = {"messages": [HumanMessage(content="Can you review my LinkedIn profile?")]}
    result = graph.invoke(state)

    while result.is_interrupted:
        user_input = input("You: ")
        state["messages"].append(HumanMessage(content=user_input))
        result = graph.resume(result, {"messages": [HumanMessage(content=user_input)]})
        state = result

    for m in result["messages"]:
        if isinstance(m, AIMessage):
            print(f"\nAI: {m.content}\n")
