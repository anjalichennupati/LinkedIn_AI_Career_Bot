import os
import json
import operator
from typing import TypedDict, Annotated, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt

from pydantic import BaseModel, Field
from typing import List
from langmem import (
    create_memory_store_manager,
    create_prompt_optimizer,
    create_manage_memory_tool,
    create_search_memory_tool
)


import os
from pymongo import MongoClient
import traceback

import asyncio
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
        print(f"Created DB '{db_name}' and collection '{collection_name}'")

    # --- LangGraph Saver ---
    memory = MongoDBSaver(
        client=client, db_name="career_bot", collection_name="checkpoints"
    )

    print("Using MongoDB persistence")

except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    tb = traceback.format_exc()
    print(f"Falling back to in-memory persistence: {e}|trace={tb}")




class CareerMemory(BaseModel):
    """Structured memory for storing insights about a user's career planning session."""
    career_goal: str = Field(description="The user's specific career objective or transition goal")
    timeline: Optional[str] = Field(description="Timeframe mentioned for achieving the goal")
    current_role: Optional[str] = Field(description="User's current position/role")
    target_role: Optional[str] = Field(description="Desired position/role")
    key_skills_needed: Optional[List[str]] = Field(description="Important skills mentioned or needed")
    learning_preferences: Optional[str] = Field(description="Preferred learning methods or platforms")
    industry_focus: Optional[str] = Field(description="Target industry or domain of interest")
    success_strategies: Optional[str] = Field(description="Approaches that have worked well for similar goals")
    common_obstacles: Optional[str] = Field(description="Challenges typically faced in this type of transition")
    resource_recommendations: Optional[str] = Field(description="Specific courses, tools, or resources that proved effective")

# LangMem setup
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

    from langgraph.store.mongodb.base import MongoDBStore
    db = client["career_bot_memories"]
    memory_store = MongoDBStore(collection=db["procedural_memories"])

    memory_store_manager = create_memory_store_manager(
        llm,
        schemas=[CareerMemory],
        store=memory_store
    )

    prompt_optimizer = create_prompt_optimizer(llm)
    manage_memory_tool = create_manage_memory_tool(memory_store_manager)
    search_memory_tool = create_search_memory_tool(memory_store_manager)

    LANGMEM_AVAILABLE = True
except Exception as e:
    print(f"❌ LangMem initialization failed: {e}")
    memory_store_manager = None
    LANGMEM_AVAILABLE = False



import google.generativeai as genai
from scraper_utils import scrape_and_clean_profile

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")





def _log_transition(thread_id: str, node: str, action: str, note: str = ""):
    try:
        logging.info(f"[flow] thread={thread_id} node={node} action={action} {note}")
    except Exception:
        pass


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    profile_data: Annotated[dict, lambda _, x: x]
    current_job_description: Annotated[Optional[str], lambda _, x: x]
    linkedin_url: Annotated[Optional[str], lambda _, x: x]
    thread_id: Annotated[Optional[str], lambda _, x: x]  
    websearch_results: Annotated[list, lambda _, x: x]
    websearch_summary: Annotated[list, lambda _, x: x]
    # FIX 1: Add flag to track if profile was already scraped
    profile_scraped: Annotated[bool, lambda _, x: x]


from apify_client import ApifyClient
import time

# FIX 3: Improved websearch with better error handling and link extraction
def websearch_mcp_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "websearch_mcp", "start")
    messages = state.get("messages", [])
    messages.append(AIMessage("🔍 Fetching web data..."))

    # extract latest human query
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content.strip()
            break

    if not query:
        messages.append(AIMessage("⚠ No query provided for web search."))
        state["messages"] = messages
        _log_transition(thread_id, "websearch_mcp", "done", "no-query")
        return state

    api_token = os.getenv("APIFY_API_KEY")
    if not api_token:
        messages.append(AIMessage("❌ Apify API token not configured."))
        state["messages"] = messages
        _log_transition(thread_id, "websearch_mcp", "done", "no-token")
        return state

    client = ApifyClient(api_token)
    actor_id = "WMg2EXLzJGPVQ5Vfq"   # âœ… stick with the tested actor

    input_data = {
        "query": query,
        "num": 1,
        "start": 1
    }

    max_retries = 3
    retry_delay = 2
    results = []

    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f"Starting Apify web search for: {query}")
            run = client.actor(actor_id).call(run_input=input_data)

            dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
            print(dataset_items)
            logging.info(f"Retrieved {len(dataset_items)} items from Apify")

            for item in dataset_items:
                result = {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "date": item.get("date")
                }
                results.append(result)

            print("printing search results:", results)


            if results:
                messages.append(
                    AIMessage(
                        f"ðŸ” WebSearch results fetched ({len(results)} results):\n"
                        + "\n".join([f"- {r['title']}: {r['link']}" for r in results[:3]])
                    )
                )
            else:
                messages.append(AIMessage("⚠ No search results found."))
            break

        except Exception as e:
            logging.error(f"Apify search attempt {attempt} failed: {e}")
            if attempt < max_retries:
                messages.append(AIMessage(f"⚠ Apify attempt {attempt} failed. Retrying..."))
                time.sleep(retry_delay)
            else:
                messages.append(AIMessage(f"❌ Apify WebSearch failed after {max_retries} attempts: {e}"))

    state["messages"] = messages
    state["websearch_results"] = results
    state["web_search"] = results
    _log_transition(thread_id, "websearch_mcp", "done", f"results={len(results)}")
    return state



def enrich_websearch_node(state):
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "enrich_websearch", "start")
    messages = state.get("messages", [])
    results = state.get("websearch_results", [])
    
    if not results:
        messages.append(AIMessage("⚠ No search results to enrich."))
        state["messages"] = messages
        _log_transition(thread_id, "enrich_websearch", "done", "no-results")
        return state

    summary = []
    for r in results:
        # Use snippet if available, otherwise try to fetch content
        content = r.get("snippet") or r.get("content") or r.get("description", "")
        
        if not content and r.get("link"):
            # Fallback to simple content fetch
            try:
                import requests
                response = requests.get(r["link"], timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                content = response.text[:2000]  # Limit content
            except Exception as e:
                content = f"Could not fetch content: {e}"

        # Summarize content
        try:
            if content and len(content) > 50:
                summary_prompt = f"Summarize the key points for career planning from this content:\n\n{content[:1500]}"
                res = llm_call_with_retry_circuit(summary_prompt)
                summary_text = res.text
            else:
                summary_text = content or "No content available"
                
            summary.append({
                "title": r["title"], 
                "link": r["link"], 
                "summary": summary_text[:500]  # Limit summary length
            })
            
        except Exception as e:
            logging.error(f"Error enriching search result: {e}")
            summary.append({
                "title": r["title"], 
                "link": r["link"], 
                "summary": f"âŒ Could not summarize: {e}"
            })

    messages.append(AIMessage(f"✅ Search results enriched with AI summaries ({len(summary)} results)."))
    state["messages"] = messages
    state["websearch_summary"] = summary
    _log_transition(thread_id, "enrich_websearch", "done", f"summary={len(summary)}")
    return state


def store_websearch_node(state: AgentState) -> dict:
    """No-op for DB. Keep websearch_summary in state only; storage happens in plan/review nodes."""
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "store_websearch", "done")
    return state





# FIX 1: Improved LinkedIn scraper that only runs when needed
def linkedin_scraper_node(state: dict) -> dict:
    """
    Scrapes LinkedIn profile URL only if not already scraped in this session.
    """
    messages = state.get("messages", [])
    thread_id = state.get("thread_id", "default_thread")
    profile_scraped = state.get("profile_scraped", False)
    existing_profile = state.get("profile_data")

    # FIX 1: Skip scraping if profile already exists and was scraped
    _log_transition(thread_id, "linkedin_scraper", "start")
    if profile_scraped and existing_profile:
        logging.info("Profile already scraped in this session, skipping scraper")
        _log_transition(thread_id, "linkedin_scraper", "goto", "career_qa_router")
        return Command(goto=["career_qa_router"], update=state)

    linkedin_url = state.get("linkedin_url", "").strip()

    if not linkedin_url:
        messages.append(AIMessage("Please provide a LinkedIn profile URL to begin."))
        state["messages"] = messages
        _log_transition(thread_id, "linkedin_scraper", "interrupt", "awaiting URL")
        return interrupt("await_input")

    messages.append(AIMessage("Scraping your LinkedIn profile..."))

    try:
        scraped_profile = scrape_and_clean_profile(
            linkedin_url=linkedin_url, api_token=os.getenv("APIFY_API_KEY")
        )

        if not scraped_profile:
            messages.append(AIMessage("Failed to extract profile. Try again."))
            state["messages"] = messages
            _log_transition(thread_id, "linkedin_scraper", "interrupt", "scrape failed")
            return interrupt("await_input")

        messages.append(AIMessage("Profile successfully scraped!"))
        state["profile_data"] = scraped_profile
        state["profile_scraped"] = True  # FIX 1: Mark as scraped
        state["messages"] = messages
        state["thread_id"] = thread_id

    except Exception as e:
        messages.append(AIMessage(f"Error scraping LinkedIn profile: {e}"))
        state["messages"] = messages
        _log_transition(thread_id, "linkedin_scraper", "interrupt", "exception during scrape")
        return interrupt("await_input")

    _log_transition(thread_id, "linkedin_scraper", "goto", "career_qa_router")
    return Command(goto=["career_qa_router"], update=state)


def career_qa_router(state: AgentState) -> Command:
    thread_id = state.get("thread_id", "default_thread")
    profile = state.get("profile_data", {})
    jd = state.get("current_job_description", "")
    messages = state.get("messages", [])
    _log_transition(thread_id, "career_qa_router", "start")
    question = ""

    # Determine last speaker and latest human question
    last_is_human = False
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_is_human = False
            break
        if isinstance(msg, HumanMessage):
            question = msg.content.strip()
            last_is_human = True
            break

    # If profile not scraped yet, route to scraper once
    if not state.get("profile_scraped") or not state.get("profile_data"):
        if state.get("linkedin_url", "").strip():
            _log_transition(thread_id, "career_qa_router", "goto", "linkedin_scraper")
            return Command(goto="linkedin_scraper", update=state)
        # Ask for URL
        messages.append(AIMessage("Please provide a LinkedIn profile URL to begin."))
        state["messages"] = messages
        _log_transition(thread_id, "career_qa_router", "interrupt", "awaiting linkedin url")
        return interrupt("await_input")

    # If last message is not from human, wait for next input
    if not last_is_human or not question:
        state["messages"] = messages
        _log_transition(thread_id, "career_qa_router", "interrupt", "awaiting next question")
        return interrupt("await_input")

    if question.lower() in {"quit", "exit", "stop"}:
        messages.append(AIMessage("Okay, ending the conversation."))
        _log_transition(thread_id, "career_qa_router", "goto", "END")
        return Command(goto=END, update={"messages": messages})

    if "job description:" in question.lower():
        messages.append(AIMessage("Got your new job description."))
        state["messages"] = messages
        state["current_job_description"] = question
        _log_transition(thread_id, "career_qa_router", "interrupt", "job description captured")
        return interrupt("await_input")

    

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
â†’ If the user wants a LinkedIn/resume/profile review, feedback, strengths, weaknesses, or audit.

Examples:
- "Can you review my LinkedIn?"
- "What are my strengths and weaknesses?"
- "Audit my profile"
- Or any other question that implies analyzing the profile.
---

ROUTE TO: job_fit_agent
â†’ If the user says anything like:
- "Does my profile match this JD?"
- "Am I eligible for this job?"
- "Score me against this role"
- Or any other question that implies matching the profile to a job description.

Only route here if a job description was recently provided.

---

ROUTE TO: enhance_profile
â†’ If the user asks for:
- Rewriting/resume improvement
- Profile optimization
- "Improve my About section"
- "Rewrite my Experience bullets"
- Or any other question that implies enhancing the profile.

---

ROUTE TO: career_plan
â†’ If the user wants a career plan, roadmap, or action plan.

Examples:
- "Create a career plan for me"
- "Give me a roadmap to become a data scientist"
- "I want a 30-60-90 day plan"
- "Help me plan my career transition"
- Or any other question that implies creating a structured career plan.

---

ROUTE TO: general_qa
â†’ All other general career questions.

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
        # Normalize common phrasing to decisions
        mapping = {
            "analyze": "analyze_profile",
            "analysis": "analyze_profile",
            "profile": "analyze_profile",
            "review": "analyze_profile",
            "job fit": "job_fit_agent",
            "fit": "job_fit_agent",
            "enhance": "enhance_profile",
            "improve": "enhance_profile",
            "plan": "career_plan",
            "roadmap": "career_plan",
            "general": "general_qa",
            "qa": "general_qa",
        }
        for key, val in mapping.items():
            if key in decision and val in {"analyze_profile","job_fit_agent","enhance_profile","career_plan","general_qa"}:
                decision = val
                break

        if decision in {
            "analyze_profile",
            "job_fit_agent",
            "enhance_profile",
            "career_plan",
            "general_qa",
        }:
            state["messages"] = messages

            _log_transition(thread_id, "career_qa_router", "goto", decision)
            return Command(
                goto=decision if decision != "general_qa" else "general_qa_node",
                update=state
            )

        else:
            messages.append(AIMessage("âš  I didn't understand. Try rephrasing."))
            state["messages"] = messages
            _log_transition(thread_id, "career_qa_router", "interrupt", "could not route")
            return interrupt()
    except Exception as e:
        messages.append(AIMessage(f"âš  Routing error: {e}"))
        state["messages"] = messages
        _log_transition(thread_id, "career_qa_router", "interrupt", "routing exception")
        return interrupt()


def analyze_profile_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "analyze_profile", "start")
    profile = state.get("profile_data")
    messages = state["messages"]
    if not profile:
        messages.append(AIMessage("âš  No profile data found to analyze."))
        state["messages"] = messages
        return interrupt()

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
        List 3â€“5 strengths that stand out across the profile.

        ## Weaknesses  
        List 3â€“5 major weaknesses holding the profile back.
 ## Section-by-Section Evaluation  
        For each section (About, Experience, Projects, Education, Skills, etc.), write:

        - Give a quality score: âœ… Strong / âš  Needs improvement / âŒ Missing  
        - Provide 2â€“3 suggestions to improve the section (content, phrasing, structure)  
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
        messages.append(AIMessage(f"âŒ Error: {e}"))
        state["messages"] = messages

    _log_transition(thread_id, "analyze_profile", "goto", "career_qa_router")
    return Command(goto="career_qa_router", update=state)


def job_fit_agent_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "job_fit_agent", "start")
    profile = state.get("profile_data")
    jd = state.get("current_job_description", "")
    messages = state["messages"]
    if not profile or not jd:
        messages.append(AIMessage("âš  Missing profile or job description."))
        state["messages"] = messages
        return interrupt()

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
        3. List 3â€“5 strengths from the candidate's profile that match the job expectations.
        4. Suggest 3â€“5 concrete improvements â€“ these could include skill gaps, experience tweaks, weak areas in phrasing, or missing proof of impact.
  5. Only evaluate against the given job role. Do not assume adjacent job titles are valid matches.
        6. If the candidate seems overqualified or underqualified, clearly state it and explain how that affects the match.

        ---

        OUTPUT FORMAT:
        # ðŸŽ¯ Job Fit Evaluation

        ## âœ… Job Match Score: XX/100
        - One-line explanation of the score.
        - 2â€“3 bullets with specific justification.

        ## ðŸŸ© Strengths
        - Point 1 (aligned with JD)
        - Point 2
        - Point 3  
        (Each point â‰¤ 40 words)

 ## ðŸŸ¥ Weaknesses
        - specify the top 3-4 points as to why this profile doesn't match the job or will get rejected even if applied and this analysis must be honest and brutal
        (Each point â‰¤ 40 words)

        ## ðŸ›  Improvements to Increase Match Score
        - Point 1 (what to improve and how)
        - Point 2
        - Point 3  
        (Each point â‰¤ 25 words)

        ## ðŸ“Œ Verdict
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
        messages.append(AIMessage(f"âŒ Error: {e}"))
        state["messages"] = messages

    _log_transition(thread_id, "job_fit_agent", "goto", "career_qa_router")
    return Command(goto="career_qa_router", update=state)


def enhance_profile_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "enhance_profile", "start")
    profile = state.get("profile_data")
    jd = state.get("current_job_description", "")
    messages = state["messages"]
    if not profile:
        messages.append(AIMessage("âš  No profile found to enhance."))
        state["messages"] = messages
        return interrupt()

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
        4. Each bullet must be â‰¤ 25 words, and **max 4 bullets per section.
        5. For the "About" section, limit to 2â€“3 tight paragraphs, total **â‰¤ 250 words.
        6. Add impactful verbs, metrics, and proof of value wherever possible.
        7. If the user requested only a section enhancement (e.g., just projects), modify only that section.
        8. If the user asks for a specific section (e.g., "improve my projects"), focus only on that section, DO NOT TALK ABOUT OTHER SECTIONS.

        FORMAT:
        Return the improved sections in clean Markdown format.
        - Use bold section titles.
        - Show only modified sections â€“ skip untouched ones to save tokens.
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
        messages.append(AIMessage(f"âŒ Error: {e}"))
        state["messages"] = messages

    _log_transition(thread_id, "enhance_profile", "goto", "career_qa_router")
    return Command(goto="career_qa_router", update=state)


def general_qa_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "general_qa_node", "start")
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
        messages.append(AIMessage(f"âŒ Error: {e}"))
        state["messages"] = messages

    _log_transition(thread_id, "general_qa_node", "goto", "career_qa_router")
    return Command(goto="career_qa_router", update=state)


# Import remaining modules as in original...
# (ThreadBasedProceduralMemory class and other imports remain the same)

# ThreadBasedProceduralMemory class - keep as is from original



# MCP client setup (keep same as original)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys

CAREER_PLAN_SERVER = os.path.abspath(
    "E:\\LinkedIn_AI_Career_Bot - Copy\\linkedin\\career_plan_mcp.py"
)

async def call_career_plan_mcp(profile_data, messages, system_prompt):
    """Async call to the career-plan MCP tool."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[CAREER_PLAN_SERVER],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "generate_career_plan",
                arguments={
                    "profile_data": profile_data,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        }
                    ] + [
                        {
                            "role": (
                                "user" if isinstance(m, HumanMessage) else "assistant"
                            ),
                            "content": m.content,
                        }
                        for m in messages
                    ],
                },
            )
            return result.content[0].text


def get_user_id_from_profile(profile: dict) -> str:
    """Extract user ID from profile"""
    return profile.get("user_id", 
           profile.get("email", 
           profile.get("linkedin_url", 
           f"user_{hash(str(profile))}")))


def extract_latest_user_query(messages: list[BaseMessage]) -> str:
    """Extract the latest user query from messages"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content.strip()
    return ""


import asyncio

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If already inside an event loop, schedule and wait
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    else:
        return asyncio.run(coro)



# FIX 2: Enhanced career plan node with proper context preservation
def career_plan(state: AgentState) -> dict:
    messages = state.get("messages", [])
    thread_id = state.get("thread_id", "unknown")

    logging.info(f"[flow] thread={thread_id} node=career_plan action=start")

    if not LANGMEM_AVAILABLE:
        messages.append(AIMessage("❌ Error: Memory system is not available."))
        state["messages"] = messages
        logging.error(f"[flow] thread={thread_id} node=career_plan action=error LangMem unavailable")
        return interrupt("await_input")

    try:
        profile_data = state.get("profile_data", {})
        user_query = extract_latest_user_query(messages)

        # Sync call to GradientPromptOptimizer (ensure it's a sync call from the library)
        optimized_prompt = asyncio.run(prompt_optimizer(  # This should now be a synchronous call
            prompt={
                "name": "career_plan_prompt",
                "prompt": """Key Instructions:
                    1. ALWAYS refer back to the conversation history
                    2. Keep focus on the role/timeline mentioned
                    3. Apply modifications to the ORIGINAL request
                    4. Be specific and actionable
                """
            },
            trajectories=[{
                "messages": [
                    msg.dict() if hasattr(msg, "dict") else {
                        "role": msg.type.lower(),
                        "content": msg.content
                    }
                    for msg in messages
                ]
            }]
        ))

        logging.info(f"[flow] thread={thread_id} node=career_plan action=prompt_optimized")

        # Call MCP career plan generator synchronously (ensure it's sync as well)
        plan_text = run_async(call_career_plan_mcp(profile_data, messages, optimized_prompt))

        logging.info(f"[flow] thread={thread_id} node=career_plan action=plan_generated")

        # Store in procedural memory (use put instead of add)
        config = {
            "langgraph_user_id": str(thread_id)  # or use str(user_id) if you're dealing with users
        }
        memory_store_manager.put(
            key="career_plan_{}".format(thread_id),  # Unique key based on thread_id
            value={"career_goal": user_query, "resource_recommendations": plan_text, "metadata": {"thread_id": thread_id}},
            config=config  # Pass the config here
        )
        logging.info(f"[flow] thread={thread_id} node=career_plan action=memory_stored")

        messages.append(AIMessage(plan_text))
        state["messages"] = messages

    except Exception as e:
        logging.error(f"[flow] thread={thread_id} node=career_plan action=error error='{e}'")
        messages.append(AIMessage(f"❌ Error generating career plan: {e}"))
        state["messages"] = messages

    logging.info(f"[flow] thread={thread_id} node=career_plan action=interrupt await_review")
    return interrupt("await_review")






def career_plan_review_node(state: AgentState) -> dict:
    """Takes the last generated plan + human review notes and modifies the plan."""
    thread_id = state.get("thread_id", "default_thread")
    _log_transition(thread_id, "career_plan_review", "start")
    messages = state.get("messages", [])
    # Find last AI plan and latest human feedback
    last_ai_plan = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    last_human_feedback = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    # Always refresh web search context for revised plan
    try:
        updated = websearch_mcp_node(state)
        updated = enrich_websearch_node(updated)
        updated = store_websearch_node(updated)
        state = updated
    except Exception:
        pass

    websearch_summary = state.get("websearch_summary", [])
    web_context = ""
    if websearch_summary:
        web_context = "\n\nWEB RESEARCH:\n" + "\n".join(
            f"- {r.get('title','')}: {r.get('link','')}" for r in websearch_summary[:5]
        )

    prompt = f"""
You are refining an existing career plan based on human review notes.

Existing plan:
{last_ai_plan}

Review notes from user:
{last_human_feedback}
{web_context}

Rewrite ONLY the plan, integrating the feedback. Keep it structured, specific, and actionable. Preserve useful content; modify where needed.
"""
    try:
        res = llm_call_with_retry_circuit(prompt)
        messages.append(AIMessage(res.text))
        state["messages"] = messages
    except Exception as e:
        messages.append(AIMessage(f"âŒ Error refining plan: {e}"))
        state["messages"] = messages

    # Pause again for further review cycles
    _log_transition(thread_id, "career_plan_review", "interrupt", "await_review")
    return interrupt("await_review")



def build_qa_graph():
    """Segment 1: Scraper â†’ Router â†’ {analyze_profile, job_fit_agent, enhance_profile, general_qa}"""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("linkedin_scraper", linkedin_scraper_node)
    builder.add_node("career_qa_router", career_qa_router)
    builder.add_node("analyze_profile", analyze_profile_node)
    builder.add_node("job_fit_agent", job_fit_agent_node)
    builder.add_node("enhance_profile", enhance_profile_node)
    builder.add_node("general_qa_node", general_qa_node)

    # Entry at router; scraper only called from router once if needed
    builder.set_entry_point("career_qa_router")

    # Scraper → router
    builder.add_edge("linkedin_scraper", "career_qa_router")

    # Router → nodes are dynamic via Command.go_to (no static fan-out edges)

    # Task nodes loop back to router; router will interrupt to await next question
    for task in ["analyze_profile", "job_fit_agent", "enhance_profile", "general_qa_node"]:
        builder.add_edge(task, "career_qa_router")

    return builder.compile(checkpointer=memory)


def build_plan_graph():
    """Segment 2: LLM decides web search â†’ (websearch_mcp â†’ enrich â†’ store)? â†’ career_plan â†’ await_review â†’ career_plan_review"""
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("websearch_mcp", websearch_mcp_node)
    builder.add_node("enrich_websearch", enrich_websearch_node)
    builder.add_node("store_websearch", store_websearch_node)
    builder.add_node("career_plan", career_plan)
    builder.add_node("career_plan_review", career_plan_review_node)

    # Entry: always do websearch before planning
    builder.set_entry_point("websearch_mcp")

    # Web search flow
    builder.add_edge("websearch_mcp", "enrich_websearch")
    builder.add_edge("enrich_websearch", "store_websearch")
    builder.add_edge("store_websearch", "career_plan")

    # Decider can route to search or directly to career_plan (done inside node via Command)
    # No explicit static edge needed; the decider returns Command to the chosen node.

    # After plan interrupts, resume continues to review
    builder.add_edge("career_plan", "career_plan_review")

    return builder.compile(checkpointer=memory)


# Compile both segments
qa_graph = build_qa_graph()
plan_graph = build_plan_graph()

all = ["qa_graph", "plan_graph", "memory", "thread_procedural_memory", "test_thread_based_learning"]





all = ["graph", "memory"]


if _name_ == "_main_":
    print("--- Testing Graph with All Fixes ---")
    
    # Test the complete workflow
    test_state = {
        "messages": [HumanMessage(content="I want to transition from software engineering to AI/ML")],
        "profile_data": {
            "headline": "Senior Software Engineer", 
            "skills": ["Python", "JavaScript", "React"],
            "experience": "5 years",
            "user_id": "test_user_integration"
        },
        "linkedin_url": "https://linkedin.com/in/testuser",
        "thread_id": "integration_test_thread",
        "profile_scraped": True  # Mark as already scraped
    }
    
    try:
        result = qa_graph.invoke(
            test_state,
            config={"configurable": {"thread_id": "integration_test_thread"}}
        )
        
        print("âœ… Graph execution successful")
        print(f"Messages generated: {len(result.get('messages', []))}")
        
    except Exception as e:
        print(f"âŒ Graph execution failed: {e}")
        traceback.print_exc()
