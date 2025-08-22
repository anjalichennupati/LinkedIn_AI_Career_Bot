import os
import json
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

import os
from pymongo import MongoClient
import traceback

import time
import functools
import logging

from dotenv import load_dotenv

load_dotenv()  # <-- MUST be at the very top before any os.getenv

APIFY_TOKEN = os.getenv("APIFY_API_KEY")  #

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Circuit breaker state
circuit_open = False
failure_count = 0
failure_threshold = 3  # Open circuit after 3 consecutive failures
circuit_reset_time = 30  # seconds
last_failure_time = 0


def llm_call_with_retry_circuit(prompt: str, max_retries=3, retry_delay=2):
    """
    Wrapper for model.generate_content with:
    - Retry on transient errors
    - Circuit breaker to stop repeated failures
    - Token/cost logging
    """
    global circuit_open, failure_count, last_failure_time

    # Circuit breaker check
    if circuit_open:
        if time.time() - last_failure_time < circuit_reset_time:
            logging.warning("Circuit open. Skipping LLM call.")
            raise RuntimeError("Circuit open due to repeated failures")
        else:
            logging.info("Resetting circuit breaker.")
            circuit_open = False
            failure_count = 0

    for attempt in range(1, max_retries + 1):
        try:
            res = model.generate_content(prompt)

            # Logging token/cost info if available
            if hasattr(res, "usage"):
                tokens = res.usage.get("total_tokens", "N/A")
                cost = tokens * 0.00001  # Example: adjust based on model pricing
                logging.info(
                    f"LLM call successful | Tokens used: {tokens} | Est. cost: ${cost:.6f}"
                )

            # Reset failure count on success
            failure_count = 0
            return res

        except Exception as e:
            logging.warning(f"LLM call failed on attempt {attempt}: {e}")
            failure_count += 1
            last_failure_time = time.time()
            if failure_count >= failure_threshold:
                circuit_open = True
                logging.error(
                    f"Circuit opened after {failure_count} consecutive failures."
                )
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")


# Try to use MongoDB persistence, fallback to memory if not available
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver

    mongo_url = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

    # --- Explicit connection check ---
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=2000)
    client.admin.command("ping")  # will raise if not connected

    db_name = "career_bot"
    collection_name = "checkpoints"
    db = client["career_bot"]
    collection = db["checkpoints"]

    # --- Force creation of DB + collection by inserting a dummy if empty ---
    if collection.count_documents({}) == 0:
        collection.insert_one({"_init": True})
        print(f"📂 Created DB '{db_name}' and collection '{collection_name}'")

    # --- LangGraph Saver ---
    memory = MongoDBSaver(
        client=client, db_name="career_bot", collection_name="checkpoints"
    )

    print("✅ Using MongoDB persistence")

except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    tb = traceback.format_exc()
    print(f"⚠ Falling back to in-memory persistence: {e}|trace={tb}")


import google.generativeai as genai
from scraper_utils import scrape_and_clean_profile

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    profile_data: Annotated[dict, lambda _, x: x]
    current_job_description: Annotated[Optional[str], lambda _, x: x]
    linkedin_url: Annotated[Optional[str], lambda _, x: x]  # <<< add this


from apify_client import ApifyClient
import time


def websearch_mcp_node(state: AgentState) -> dict:
    messages = state["messages"]
    messages.append(AIMessage("🔍 Fetching web data..."))

    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content.strip()
            break

    if not query:
        messages.append(AIMessage("⚠ No query provided for web search."))
        state["messages"] = messages
        return interrupt("continue_chat")

    api_token = os.getenv("APIFY_API_TOKEN")
    client = ApifyClient(api_token)

    actor_id = "desearch~web-search"
    # replace with your Apify actor ID for web search
    input_data = {"query": query, "maxResults": 5}

    max_retries = 3
    retry_delay = 2
    results = []

    for attempt in range(1, max_retries + 1):
        try:
            run = client.actor(actor_id).call(run_input=input_data)
            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            for item in dataset_items:
                results.append({"title": item.get("title"), "link": item.get("url")})

            messages.append(
                AIMessage(
                    "🔍 WebSearch results fetched via Apify:\n"
                    + "\n".join([f"- {r['title']}: {r['link']}" for r in results])
                )
            )
            break
        except Exception as e:
            if attempt < max_retries:
                messages.append(
                    AIMessage(f"⚠ Apify attempt {attempt} failed. Retrying...")
                )
                time.sleep(retry_delay)
            else:
                messages.append(
                    AIMessage(
                        f"❌ Apify WebSearch failed after {max_retries} attempts: {e}"
                    )
                )

    state["messages"] = messages
    state["websearch_results"] = results
    return interrupt("continue_chat")


def enrich_websearch_node(state: AgentState) -> dict:
    messages = state["messages"]
    messages.append(AIMessage("✅ Web results enriched."))

    results = state.get("websearch_results", [])

    if not results:
        messages.append(AIMessage("⚠ No search results to enrich."))
        state["messages"] = messages
        return interrupt("continue_chat")

    summary = []
    for r in results:
        prompt = f"Summarize the key points from this webpage: {r['link']}"
        try:
            res = llm_call_with_retry_circuit(prompt)

            summary.append(
                {"title": r["title"], "link": r["link"], "summary": res.text}
            )
        except Exception as e:
            summary.append(
                {"title": r["title"], "link": r["link"], "summary": f"❌ Error: {e}"}
            )

    messages.append(AIMessage("✅ Search results enriched with AI summaries."))
    state["messages"] = messages
    state["websearch_summary"] = summary
    return interrupt("continue_chat")


def store_websearch_node(state: AgentState) -> dict:
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = client["career_bot"]
    collection = db["websearch"]

    results = state.get("websearch_summary", [])
    if results:
        collection.insert_many(results)
        state["messages"].append(
            AIMessage(f"💾 Stored {len(results)} search results in DB.")
        )
    else:
        state["messages"].append(AIMessage("⚠ Nothing to store."))

    return interrupt("continue_chat")


import os
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage
from scraper_utils import scrape_and_clean_profile


# linkedin_scraper_node.py
from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage
from scraper_utils import scrape_and_clean_profile
import os


def linkedin_scraper_node(state: dict) -> dict:
    """
    Scrapes LinkedIn profile URL from state['linkedin_url'] and stores result in state['profile_data'].
    """
    messages = state.get("messages", [])

    linkedin_url = state.get("linkedin_url", "").strip()

    if not linkedin_url:
        messages.append(AIMessage("⚠ Please provide a LinkedIn profile URL to begin."))
        state["messages"] = messages
        return interrupt("continue_chat")

    messages.append(AIMessage("🔎 Scraping your LinkedIn profile..."))

    try:
        scraped_profile = scrape_and_clean_profile(
            linkedin_url=linkedin_url, api_token=os.getenv("APIFY_API_KEY")
        )

        if not scraped_profile:
            messages.append(AIMessage("❌ Failed to extract profile. Try again."))
            state["messages"] = messages
            # return interrupt("continue_chat")

        messages.append(AIMessage("✅ Profile successfully scraped!"))
        state["profile_data"] = scraped_profile
        state["messages"] = messages

    except Exception as e:
        messages.append(AIMessage(f"❌ Error scraping LinkedIn profile: {e}"))
        state["messages"] = messages

    # return interrupt("continue_chat")  # <-- that's all you need
    return Command(goto=["career_plan", "career_qa_router"], update=state)


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
- career_plan
- general_qa

DO NOT GUESS. Use the following rules:

---

ROUTE TO: analyze_profile
→ If the user wants a LinkedIn/resume/profile review, feedback, strengths, weaknesses, or audit.

Examples:
- "Can you review my LinkedIn?"
- "What are my strengths and weaknesses?"
- "Audit my profile"
- Or any other question that implies analyzing the profile.
---

ROUTE TO: job_fit_agent
→ If the user says anything like:
- "Does my profile match this JD?"
- "Am I eligible for this job?"
- "Score me against this role"
- Or any other question that implies matching the profile to a job description.

Only route here if a job description was recently provided.

---

ROUTE TO: enhance_profile
→ If the user asks for:
- Rewriting/resume improvement
- Profile optimization
- "Improve my About section"
- "Rewrite my Experience bullets"
- Or any other question that implies enhancing the profile.

---

ROUTE TO: career_plan
→ If the user wants a career plan, roadmap, or action plan.

Examples:
- "Create a career plan for me"
- "Give me a roadmap to become a data scientist"
- "I want a 30-60-90 day plan"
- "Help me plan my career transition"
- Or any other question that implies creating a structured career plan.

---

ROUTE TO: general_qa
→ All other general career questions.

Examples:
- "What kind of roles should I target?"
- "How do I switch fields?"
- "What are good certifications for data science?"
- "How do I get into startups?"
- Or any other question that doesn't fit the above categories.

---

USER QUESTION:
{question}

Just respond with ONE of:
analyze_profile, job_fit_agent, enhance_profile, career_plan, general_qa
"""

    try:
        result = llm_call_with_retry_circuit(prompt)
        decision = result.text.strip().lower()

        if decision in {
            "analyze_profile",
            "job_fit_agent",
            "enhance_profile",
            "career_plan",
            "general_qa",
        }:
            state["messages"] = messages

            return Command(
                goto=decision if decision != "general_qa" else "general_qa_node"
            )

        else:
            messages.append(AIMessage("⚠ I didn’t understand. Try rephrasing."))
            state["messages"] = messages
            return interrupt("continue_chat")
    except Exception as e:
        messages.append(AIMessage(f"⚠ Routing error: {e}"))
        state["messages"] = messages
        return interrupt("continue_chat")


# 🔍 analyze_profile_node
def analyze_profile_node(state: AgentState) -> dict:
    profile = state.get("profile_data")
    messages = state["messages"]
    if not profile:
        messages.append(AIMessage("⚠ No profile data found to analyze."))
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
        - Max 4 bullet points per section  
        - Each bullet: <30 words  
        - Total section feedback: <100 words

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
        res = llm_call_with_retry_circuit(prompt)
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
        messages.append(AIMessage("⚠ Missing profile or job description."))
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
        1. Evaluate the fitness of the candidate for this job role using industry-standard evaluation practices (skills, experience, keywords, impact, achievements, and alignment).
        2. Return a Job Match Score out of 100, and explain how you arrived at it with specific reasoning.
        3. List 3–5 strengths from the candidate’s profile that match the job expectations.
        4. Suggest 3–5 concrete improvements — these could include skill gaps, experience tweaks, weak areas in phrasing, or missing proof of impact.
  5. Only evaluate against the given job role. Do not assume adjacent job titles are valid matches.
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
        res = llm_call_with_retry_circuit(prompt)

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
        messages.append(AIMessage("⚠ No profile found to enhance."))
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
        3. Use bullet points only where appropriate (e.g., Experience, Projects).
        4. Each bullet must be ≤ 25 words, and **max 4 bullets per section.
        5. For the “About” section, limit to 2–3 tight paragraphs, total **≤ 250 words.
        6. Add impactful verbs, metrics, and proof of value wherever possible.
        7. If the user requested only a section enhancement (e.g., just projects), modify only that section.
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
        res = llm_call_with_retry_circuit(prompt)

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

Your task is to answer general career-related questions from users

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

- Answer clearly and concisely.
- Prioritize useful, actionable advice.
- If the question is vague or broad, ask a clarifying follow-up.
- Keep the tone supportive but professional.
- Do not suggest uploading a resume or LinkedIn again.
- If you detect the user is stressed, confused, or unsure, acknowledge that supportively.

---

### Output Format:

Respond in clean text. Use bullet points or short paragraphs where needed.

Start now.
""".strip()

    try:
        res = llm_call_with_retry_circuit(prompt)

        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"❌ Error: {e}"))
        state["messages"] = messages

    return interrupt("continue_chat")


def career_plan_node(state: AgentState) -> dict:
    profile = state.get("profile_data", {})
    messages = state["messages"]
    print(state["profile_data"])

    # Find the latest user message
    latest_user_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content.strip()
            break

    if not latest_user_msg:
        messages.append(
            AIMessage("⚠ Please provide a brief or question for your career plan.")
        )
        state["messages"] = messages
        return interrupt("continue_chat")

    # Prepare profile text
    profile_text = (
        "\n".join(f"{k}: {v}" for k, v in profile.items())
        if profile
        else "(no profile)"
    )

    # Step 1: Ask the LLM if external web info is needed
    web_check_prompt = f"""
User Input: {latest_user_msg}

Profile Data:
{profile_text}

TASK:
Decide whether fetching external web info (like courses, communities, websites, industry resources) is required.  
Return JSON: {{ "use_web": true/false, "confidence": 0-100, "reason": "..." }}  
Consider the prompt’s intent, not just keywords. Base decision on whether external sources would improve the answer.
"""

    try:
        web_decision = (
            llm_call_with_retry_circuit(web_check_prompt).text.strip().lower()
        )
    except Exception as e:
        web_decision = "no"
        messages.append(
            AIMessage(f"⚠ Could not determine if web search is needed: {e}")
        )

    # Step 2: If yes, call the MCP nodes to fetch and enrich web search results
    search_summaries = ""
    if web_decision == "yes":
        messages.append(
            AIMessage("🔍 Fetching relevant web information for your career plan...")
        )
        try:
            state = websearch_mcp_node(state)
            state = enrich_websearch_node(state)
            state = store_websearch_node(state)

            # Prepare search summary to feed back into the career plan
            search_results = state.get("websearch_summary", [])
            if search_results:
                search_summaries = "\n".join(
                    f"- {r['title']}: {r['link']}" for r in search_results
                )
        except Exception as e:
            messages.append(AIMessage(f"❌ Error fetching web results: {e}"))

    # Step 3: Generate the career plan, integrating web results if any
    prompt = f"""
You are an AI career coach.

User Input: {latest_user_msg}

Profile Data:
{profile_text}

{"Additional Web Info:\n" + search_summaries if search_summaries else ""}

TASK:
- Generate a clear, actionable career plan or revise an existing one based on user input.
- Integrate relevant insights from web results if available.
- Provide practical steps, resources, and guidance.

IMPORTANT: Do NOT rely on keyword matching. Understand the user's intent from the message.
Respond concisely and practically.
""".strip()

    try:
        result = llm_call_with_retry_circuit(prompt)

        plan_output = result.text
    except Exception as e:
        plan_output = f"❌ Error generating career plan: {e}"

    messages.append(AIMessage(plan_output))
    state["messages"] = messages

    return interrupt("continue_chat")


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("linkedin_scraper", linkedin_scraper_node)

    builder.add_node("career_qa_router", career_qa_router)
    builder.add_node("analyze_profile", analyze_profile_node)
    builder.add_node("job_fit_agent", job_fit_agent_node)
    builder.add_node("enhance_profile", enhance_profile_node)
    builder.add_node("general_qa_node", general_qa_node)
    builder.add_node("career_plan", career_plan_node)

    builder.add_node("websearch_mcp", websearch_mcp_node)
    builder.add_node("enrich_websearch", enrich_websearch_node)
    builder.add_node("store_websearch", store_websearch_node)

    # Flow: fetch → enrich → store → router
    builder.add_edge("linkedin_scraper", "career_qa_router")
    builder.add_edge("websearch_mcp", "enrich_websearch")
    builder.add_edge("enrich_websearch", "store_websearch")
    builder.add_edge("store_websearch", "career_qa_router")

    builder.set_entry_point("linkedin_scraper")

    # Loop back from all task nodes to router
    for task in [
        "analyze_profile",
        "job_fit_agent",
        "enhance_profile",
        "general_qa_node",
        "career_plan",
    ]:
        builder.add_edge(task, "career_qa_router")

    return builder.compile(checkpointer=memory)


thread_id = "linkedin_bot_1"  # keep it constant per user/session


graph = build_graph()

if __name__ == "__main__":
    # State only contains messages from the front end
    state = {"messages": [HumanMessage(content="Can you review my LinkedIn profile?")]}

    # `linkedin_url` should be set dynamically from user input in the front end
    # e.g., when the user submits the URL, you do: state["linkedin_url"] = user_url_from_frontend

    result = graph.invoke(state, thread_id=thread_id)

    # Debug: see if scraper pulled the profile
    print("[DEBUG] Profile data after invoke:", state.get("profile_data"))

    while result.is_interrupted:
        # Update state first to include all prior node updates
        state = result.state

        user_input = input("You: ")
        state["messages"].append(HumanMessage(content=user_input))

        result = graph.resume(result, state)

        # Now print the latest profile_data
        print("[DEBUG] Updated profile data:", state.get("profile_data"))
