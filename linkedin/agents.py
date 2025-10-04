import os
import json
import operator
import logging
import time
from typing import TypedDict, Annotated, Optional, Dict, List, Any, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langchain_core.tools import tool


from pymongo import MongoClient
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
from apify_client import ApifyClient

# LangSmith Integration
from langsmith import Client

# strcutured output part
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from pathlib import Path


load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "linkedin_bot"
# Setup logging - Console only for global messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Module logger for existing log statements throughout the code
logger = logging.getLogger(__name__)
# Global logger for code-level changes (console only)
global_logger = logging.getLogger("global")

# Route all thread-related logs to per-thread files and hide them from console
import re
import contextvars
from typing import Optional

# Regex to detect thread ids in log messages (e.g., career-1234abcd)
_THREAD_ID_REGEX = re.compile(r"career-[0-9a-fA-F]{8}")
# Context var that marks the active thread id for logging
_current_thread_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_thread_id", default=None)

def set_thread_logging_context(thread_id: str) -> None:
    _current_thread_id.set(thread_id)

def clear_thread_logging_context() -> None:
    _current_thread_id.set(None)

class ThreadConsoleFilter(logging.Filter):
    """Filter that removes thread-related records from console output.
    Any log message containing a thread id like career-XXXXXXXX will be filtered out.
    """
    def filter(self, record: logging.LogRecord) -> bool:  # True => allow
        try:
            message = record.getMessage()
        except Exception:
            return True
        # Hide if we are inside a thread logging context
        if _current_thread_id.get() is not None:
            return False
        # Hide if the message itself mentions a thread id
        return _THREAD_ID_REGEX.search(message) is None

class PerThreadFileRouter(logging.Handler):
    """Handler that routes records to logs/{thread_id}.log based on message content.
    This preserves all existing logging calls while ensuring per-thread file isolation.
    """
    def __init__(self) -> None:
        super().__init__()
        self._handlers_by_thread: dict[str, logging.FileHandler] = {}
        self._logs_dir = Path("logs")
        self._logs_dir.mkdir(exist_ok=True)

    def _extract_thread_id(self, record: logging.LogRecord) -> Optional[str]:
        # Prefer context var if set
        ctx_tid = _current_thread_id.get()
        if ctx_tid:
            return ctx_tid
        try:
            message = record.getMessage()
        except Exception:
            return None
        match = _THREAD_ID_REGEX.search(message)
        return match.group(0) if match else None

    def _get_thread_handler(self, thread_id: str) -> logging.FileHandler:
        handler = self._handlers_by_thread.get(thread_id)
        if handler is None:
            log_file = self._logs_dir / f"{thread_id}.log"
            handler = logging.FileHandler(log_file, mode="a")
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self._handlers_by_thread[thread_id] = handler
            # Announce in console (global) that we created the file
            global_logger.info(f"Thread {thread_id} -> logging to logs/{thread_id}.log")
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        thread_id = self._extract_thread_id(record)
        if not thread_id:
            return
        try:
            handler = self._get_thread_handler(thread_id)
            handler.emit(record)
        except Exception:
            self.handleError(record)

# Attach console filter to all stream handlers configured by basicConfig
for h in logging.getLogger().handlers:
    if isinstance(h, logging.StreamHandler):
        h.addFilter(ThreadConsoleFilter())

# Add the router so that any logger.* call that mentions a thread id is captured to file
logging.getLogger().addHandler(PerThreadFileRouter())

# Per-thread logging setup (explicit logger retrieval for targeted logging where thread_id is known)
from pathlib import Path

# Dictionary to store thread-specific loggers
thread_loggers = {}

def setup_thread_logging(thread_id: str):
    """Setup file logging for a specific thread_id - logs ONLY to file, not console"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file path
    log_file = logs_dir / f"{thread_id}.log"
    
    # Check if logger already exists for this thread
    if thread_id not in thread_loggers:
        # Create a thread-specific logger
        thread_logger = logging.getLogger(f"thread_{thread_id}")
        thread_logger.setLevel(logging.INFO)
        
        # IMPORTANT: Prevent propagation to root logger to avoid console output
        thread_logger.propagate = False
        
        # Create file handler for this thread
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # Use the same format as console logging
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # Add handler to the thread-specific logger (NOT the root logger)
        thread_logger.addHandler(file_handler)
        
        # Store the logger reference
        thread_loggers[thread_id] = thread_logger
        
        # Log to console that a new thread log file was created
        global_logger.info(f"Thread {thread_id} -> logging to logs/{thread_id}.log")

def get_thread_logger(thread_id: str):
    """Get a logger instance for a specific thread - logs ONLY to file"""
    setup_thread_logging(thread_id)
    return thread_loggers[thread_id]

# Configure LangSmith for tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "career-assistant-agent"
# You need to set your LANGSMITH_API_KEY in your .env file
# Get it from https://smith.langchain.com/ after creating an account

# Configure Gemini with LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    stream=True,
    streaming_chunk_size=1000,
    _should_stream=True,
)

# Configure Gemini for direct calls (backup)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")


def llm_call_with_retry_circuit(prompt: str, max_retries=3, retry_delay=2):
    """Enhanced LLM call with circuit breaker and retry logic"""
    logger.info("Starting LLM call")

    for attempt in range(1, max_retries + 1):
        try:
            res = model.generate_content(prompt)
            if hasattr(res, "usage"):
                tokens = res.usage.get("total_tokens", "N/A")
                logger.info(f"LLM call successful | Tokens used: {tokens}")

            return res

        except Exception as e:
            logger.warning(f"LLM call failed on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")


# MongoDB setup - THREE COLLECTIONS STRUCTURE
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver

    mongo_url = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")

    db_name = "career_bot"
    db = client[db_name]

    # THREE COLLECTIONS
    profiles_collection = db["profiles"]  # Stores profile data
    career_plan_collection = db["career_plan"]  # Stores career plans
    conversations_collection = db["conversations"]  # Stores conversations

    # Initialize MongoDB saver for LangGraph checkpointing
    memory = MongoDBSaver(client=client, db_name=db_name, collection_name="checkpoints")
    global_logger.info("MongoDB persistence initialized with three collections")

except Exception as e:
    from langgraph.checkpoint.memory import MemorySaver

    memory = MemorySaver()
    profiles_collection = None
    career_plan_collection = None
    conversations_collection = None
    global_logger.warning(f"Falling back to in-memory persistence: {e}")

# State definition - UPDATED WITH profile_limitations
# In agents.py, update AgentState
# --- Find this class definition in agents.py and MODIFY it ---


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], operator.add]
    thread_id: str

    # --- ADD THIS LINE ---
    profile_data: Optional[Dict[str, Any]]
    career_plan: Optional[str]
    job_description: Optional[str]
    desired_job_role: Optional[str] = ""
    profile_limitations: Optional[str]

    # new changes
    simple: bool #for simple/complex
    analysis: bool #for analysis
    status: bool #for status

    # Fields for the Unified Router to manage its state machine
    pending_tasks: Optional[List[str]]
    task_results: Optional[Dict[str, Any]]
    is_planning: bool
    task_in_progress: Optional[str]

    # Recent context for router decision making
    # recent_ai_response: Optional[str]
    # previous_user_message: Optional[str]
    
    # Follow-up question for conversational flow
    follow_up: Optional[str]
    
    # Two-step flow control flags
      # True = analysis must run, False = analysis done

# structured output schemas for the nodes
# struct output for simple career plan
class SimpleCareerPlan(BaseModel):
    """The simple career plan the career_plan_node must follow which is a high level, minimal information career plan related details, which is basically a list of things"""

    lessons: List[str] = Field(
        description="""A simple career plan that outlines the key lessons or steps the user should follow to achieve their career goals in that specified timeline and split. If nothing is mentioned it should be assumed as a 3 month timeline (IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.), Each lesson must not be more than 20-30 words each.
        IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.
        """
    )
    duration: str = Field(
        description="""Duration of the detailed career plan in days, months, or years as specified by the user.  
        IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY (MUST FOLLOW THIS RULE STRICTLY).
        LAST PRIORITY: If nothing is specified, 3 months timeline.
        """
    )
    projects: List[str] = Field(
        description="List of projects that the user has or plans to work on, from their profile, else suggest 1-2 new project ideas that align with their desired career path. Should be under 40 words total limit only."
    )
    skills: List[str] = Field(
        description="List of skills that the user has or plans to work on, from their profile, else suggest 1-2 new skill ideas that align with their desired career path"
    )
    certifications: List[str] = Field(
        description="List of certifications that the user has or plans to work on, from their profile, else suggest 1-2 new certification ideas that align with their desired career path"
    )
    networking: List[str] = Field(
        description="List of networking tactics that the user needs to work on, from their profile, suggest 1-2 new networking tactic ideas that align with their desired career path"
    )


# struct output for career plan node - detailed career plan
class CareerPlan(BaseModel):
    """The detailed career plan the career_plan_node must follow, that is a comprehensive, step-by-step career plan related details as a json object containing the detailed career plan. 
    **EVERYTHING MENTIONED IN THE SCHEMA BELOW MUST BE STRICTLY FOLLOWED**
    """

    lessons: List[str] = Field(
        description="""
    A concise project description that outlines the key lessons with respect to the timeline. MUST BE IN SECOND PERSON ONLY(you, yours, you etc).
    STRICT PATTERN, MUST FOLLOW RULE:
    The lesson picked, must very specific and of high importance in that field of work.
    **[Lesson name (3-5 words ONLY)(timeline split)]** - [Explain what the lesson is in a crisp, concise and to the point format(7-9 words ONLY)] 
    MUST USE THIS PATTERN AND HEADING:
    **Personalized**: [Your personalized explanation of the lesson, MUST be personalized to the user's profile to acheive the career goals, mention which skill/project/publication/experience matches and helps for this lesson. (14-16 words ONLY). The answer must be honest.]
    Default, 3 lessons must be included. THIS SPLIT IS DYNAMIC TO WHAT THE USER IS ASKING FOR AND ALWAYS MUST BE TAILORED TO THAT. THE USER'S REQUEST ONLY MUST DETERMINE WHAT THIS SPLIT MUST BE AND YOU MUST STRICTLY ADHERE TO THAT SPLIT ASKED BY THE USER ONLY.
    IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.
    This varies by the timeline and split specified by the user.
    """
    )
    duration: str = Field(
        description="""Duration of the detailed career plan in days, months, or years as specified by the user.  
        IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY (MUST FOLLOW THIS RULE STRICTLY).
        LAST PRIORITY: If nothing is specified, 3 months timeline.
        """
    )

    projects: List[str] = Field(
        description="""
    A concise project description that outlines the key projects. MUST BE IN SECOND PERSON ONLY(you, yours, you etc
    STRICT PATTERN, MUST FOLLOW RULE:
    The project picked, must very specific and of high importance and relevance in that field of work.
    **[Project name (3-5 words ONLY)]** - [Explain what the project is and its relevance in a crisp, concise and to the point format(7-9 words ONLY) ] 
    MUST USE THIS PATTERN AND HEADING:
    **Personalized**: [Your personalized explanation of the project. First preference - must use an already existing project/skill/publication of the user to build on. Second preference - must suggest a project that must be in relevance with both- the user's profile and the career goal. (14-16 words ONLY). The answer must be honest.]
    Default, 3 projects must be included. If more are needed, then the user will ask for it.
    IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.
    This varies by the timeline and split specified by the user.
    """
    )

    skills: List[str] = Field(
        description="""
        
        A concise project description that outlines the key skills. MUST BE IN SECOND PERSON ONLY(you, yours, you etc
        STRICT PATTERN, MUST FOLLOW RULE:
        The skill picked, must very specific and of high importance and relevance in that field of work.
        **[Skill name (3-5 words ONLY)]** - [Explain what the skill is and its relevance in a crisp, concise and to the point format(7-9 words ONLY)] 
        MUST USE THIS PATTERN AND HEADING:
        **Personalized**: [Your personalized explanation of the skill. First preference - must use an already existing skill of the user to build on. Second preference - must suggest a skill that must be in relevance with both- the user's profile and the career goal. (14-16 words ONLY). The answer must be honest.]
        Default, 3 skills must be included. If more are needed, then the user will ask for it.
        IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.
        This varies by the timeline and split specified by the user.
        """
    )

    certifications: List[str] = Field(
        description="""
        A concise project description that outlines the key skills. MUST BE IN SECOND PERSON ONLY(you, yours, you etc
        STRICT PATTERN, MUST FOLLOW RULE:
        The certification picked, must very specific and of high importance and relevance in that field of work.
        **[Skill name (3-5 words ONLY)]** - [Explain what the certification is and its relevance in a crisp, concise and to the point format(7-9 words ONLY)] 
        MUST USE THIS PATTERN AND HEADING:
        **Personalized**: [Your personalized explanation of the certification. First preference - must use an already existing skill of the user to build on. Second preference - must suggest a certification that must be in relevance with both- the user's profile and the career goal. (14-16 words ONLY). The answer must be honest.]
        Default, 2 certification must be included. If more are needed, then the user will ask for it.
        IMPORTANT RULE: THE USER'S SPECIFIED TIMELINE MUST BE FIRST PRIORITY.
        This varies by the timeline and split specified by the user.
        """
    )

    networking: List[str] = Field(
        description="""
        A concise project description that outlines the key skills. MUST BE IN SECOND PERSON ONLY(you, yours, you etc
        STRICT PATTERN, MUST FOLLOW RULE:
        The networking advice picked, must very specific and of high importance and relevance.
        **[Networking tip(3-5 words ONLY)]** - [Explain its relevance in a crisp, concise and to the point format(7-9 words ONLY)] 
        MUST USE THIS PATTERN AND HEADING:
        **Personalized**: [How this networking advice can help the user achieve their career goals, how can that increase the user's chances of achieving their career goals. (14-16 words ONLY). The answer must be honest.]
        Default, 2 tips must be included. If more are needed, then the user will ask for it.
        """
    )

    web_links: List[str] = Field(
        description="This is a compulsary field that must be filled. This must be the results from the web search, conducted by the web_search_tool that are relevant to the user's career goals, It must have a clickable link to the source. If no web search was done, this field should be empty."
    )


#analysis schema from the career plan node
class Analysis(BaseModel):
    """
    This schema is for the career plan node to follow, when the analysis flag is True to perform the analysis. The complete analysis must be strcitly between 60-65 words ONLY. 
    """
    aligns: List[str] = Field(description=
    """
    The most relevant list of personalized alignments of the user's profile for the desired job role. Must mention exactly 2-4 such alignments. The total length of the output must strictly be 25-30 words ONLY. 
    """)
    lacks: List[str] = Field(description=
    """
    The most relevant list of personalized lacks of the user's profile for the desired job role. Must mention exactly 2-4 such lacks. The total length of the output must strictly be 25-30 words ONLY. 
    """)
    
class FinalOutput(BaseModel):
    """
    This schema must be STRICTLY followed by the synthesis node.
    """
    intro: str = Field(description="""
    - IMPORTANT RULE: But, IF the final response already has a professional start, YOU MUST FIRST LOOK AT THE output, if it has a professional start, then YOU MUST NOT CHANGE IT/ADD TO IT. 
    - ALWAYS NOTICE: There MUST ONLY BE 1 PROFESSIONAL START TO THE RESPONSE.
    - The final response you generate must have a professional start to it, rather than just dumping the result of the internal tasks. This must be stuck to 20-30 words ONLY. Before generating this, always understand the context of the user's request and the results of the internal tasks and generate a professional response to the user's request. This is the only change you are allowed to make, nothing related to the internal tasks, keep this in mind.
    - Always make sure while generating this professional start that within this, the user has maximum context of the results being given to them. If you think anything is missing always refer to the state context that is given to you.
    """)

    output: List[str] = Field(description= """

    CRITICALLY IMPORTANT INSTRUCTION:
    1. YOU HAVE 2 JOBS - PRINT AND COMBINE THE RESULTS OF INTERNAL TASKS, WITH NO CHANGES TO BE MADE, AND GENERATE A FOLLOW-UP QUESTION (as mentioned in the follow_up schema), THATS IT. 


    INSTRUCTIONS FOR COMBINING RESULTS:
    - If there is only 1 task result (for any of the task results), then print it AS IT IS, NO CHANGES TO BE MADE (COMPULSARY RULE).
    - COMPULSARY FOLLOW: Do not add or remove any information from the task results. Your only job is to combine the results ONLY.
    - Your final answer MUST directly address the user's original question.
    - Combine the pieces of information from the task results into a single, natural-sounding answer, NO CHANGES TO BE MADE.

    
    - The output must be the result of the internal tasks. You must not change anything from the result of the internal tasks. You must print them as they are.
    - IMPORTANT RULE: ANY OF THE SCHEMA BASED TASK RESULTS I.E ESPECIALLY FROM CAREER PLAN NODE, JOB FIT ANALYSIS NODE ETC. YOU DO NOT HAVE THE RIGHT TO CHANGE ANY RESULTS PROVIDED BY THESE NODES. YOU MUST PRINT THEM AS THEY ARE.
    - Use Markdown headings (e.g., "### Profile Enhancement") to separate different parts of the response if multiple tasks were performed.
    - Do not mention the internal tasks by name (e.g., "the enhance_profile task said...").
    - If a task failed, explain the failure clearly and politely to the user.
    - If the results include web search data, format it nicely under a 'Sources from the Web' heading with clickable Markdown links.
    - MOST IMPORTANT MUST FOLLOW RULE: THE OUTPUT MUST NEVER BE RETURNED IN JSON FORMAT. IT SHOULD ALWAYS BE A WELL FORMATTED RESPONSE. YOU MUST MAKE SURE OF THIS.
    """
    )

    follow_up: str = Field(description= """
    CRITICAL REQUIREMENT - FOLLOW-UP QUESTION:
After providing the synthesized answer, you MUST include exactly one follow-up question that:
1. Is strictly related to the output just generated (the task results) and must have a STRICT WORD LIMIT OF 8-10 WORDS ONLY.
2. Questions about actions supported by the current system ONLY:
   - Deeper skill advice or exploration (only if the user is at a simple plan stage)
   - Project exploration or enhancement (only if the user is at a simple plan stage)
   - Certification suggestions
   - Career plan modifications or finalization
   - Profile improvements
   - Job fit analysis refinements
   - General career guidance
3. Never asks about functionality outside of the implemented nodes
4. Uses natural, conversational phrasing
5. HOWEVER, the user might be asking a random question about anything in the world. And after receiving the respone you must assess very strictly that the respone to that question is in a very professional tone that is answered to the best interest of the user. BUT, YOUR FOLLOW UP QUESTION IN THIS CASE MUST ALWAYS STICK TO CAREER QUIDANCE ONLY. YOU MUST DIGRESS THE CONVERSATION BACK INTO THE VICINITY OF THIS SYSTEM'S FUNCTIONALITY MENTIONED ABOVE ONLY. 

IMPORTANT RULE MUST FOLLOW RULE FOR FOLLOW-UP QUESTION:
- Always if the analysis status is True and career plan generation status is False (it means that the user just has gotten a short analysis and is yet to get a career plan), then you MUST ask a follow-up question about whether the user wants a simple career plan.
- Always if the analysis status is True and career plan generation status is True (it means that the user is at a career plan stage), then you MUST check the career plan status if it is True, then you MUST ask a follow-up question about whether the user wants a detailed career plan.
    """
    )

#     web_links: List[str]=Field(..., description="""## Sources from the Web (if web search was used)
# - IMPORTANT: If you performed a web search, you MUST synthesize the results here. Do not mention internal instructions.
# - Format the output as a Markdown list with clickable links.
# - Example:
# - [Top Skills for Marketing Managers in 2025](URL) - This article highlights the growing importance of data analytics and AI in marketing, suggesting a focus on tools like Google Analytics and HubSpot.
# """)


# class CareerPlanCombined(BaseModel):
#     career_plan: Union[SimpleCareerPlan, CareerPlan]

class JobFitAnalysis(BaseModel):
    """The job fit analysis schema that the job_fit_agent must strictly follow, this is a comprehensive, in detail word limited schema that must be adhered to.
    """
    job_fit_score: str = Field(description="""
        ##Overall Match Score: [x/100], Provide a numerical score. The justification MUST explain the reasoning based on the weight of strengths vs. gaps.
        Keep this crisp and concise. [40-45 words ONLY]
    """)
    key_strengths: List[str] = Field(description="""
        - MOST RELEVANT list of 2-3 key strengths [3-5 words each ONLY]: [Must have this Personlized explanation for each strength] For each strength, provide a direct quote or summary from the job description's requirements and then cite the specific project, skill, or experience from the candidate's profile that serves as evidence [Each must be 10-12 words ONLY].
    """)
    critical_gaps: List[str] = Field(description="""
        - MOST RELEVANT list of 2-4 critical gaps [3-5 words each ONLY]: [Must have this Personlized explanation for each gap] For each gap, suggest a specific, actionable mitigation strategy the candidate can use in their cover letter or interview. [Each must be 12-15 words ONLY].
    """)
    final_verdict: str = Field(description="""
        - ## Final Recommendation: [The FINAL Verdit in terms of fit (in 4-5 words ONLY)] - Provide a final one-sentence recommendation. [10-12 words ONLY]
    """)



def _log_transition(thread_id: str, node: str, action: str, note: str = ""):
    """Enhanced logging for flow tracking"""
    logger.info(f"[FLOW] thread={thread_id} | node={node} | action={action} | {note}")


# MONGODB STORAGE FUNCTIONS - MODIFIED FOR PROFILES WITH LINKEDIN_URL AND JOB_DESCRIPTION
def store_profile_data(
    thread_id: str,
    profile_data: dict,
    linkedin_url: str = None,
    job_description: str = None,
):
    """Store profile data in profiles collection with linkedin_url and job_description"""
    if profiles_collection is None:
        logger.warning("MongoDB not available, skipping profile storage")
        return

    try:
        # Get existing document to preserve fields
        existing_doc = profiles_collection.find_one({"thread_id": thread_id})

        profile_doc = {
            "thread_id": thread_id,
            "profile_data": profile_data,
            "updated_at": time.time(),
        }

        # Preserve or set linkedin_url
        if linkedin_url:
            profile_doc["linkedin_url"] = linkedin_url
        elif existing_doc and existing_doc.get("linkedin_url"):
            profile_doc["linkedin_url"] = existing_doc["linkedin_url"]

        # Preserve or set job_description
        if job_description:
            profile_doc["job_description"] = job_description
        elif existing_doc and existing_doc.get("job_description"):
            profile_doc["job_description"] = existing_doc["job_description"]

        # Set created_at only if new document
        if existing_doc:
            profile_doc["created_at"] = existing_doc.get("created_at", time.time())
        else:
            profile_doc["created_at"] = time.time()

        profiles_collection.replace_one(
            {"thread_id": thread_id}, profile_doc, upsert=True
        )

        _log_transition(thread_id, "analyze_profile", "scrape_success", f"Profile data stored successfully for thread {thread_id}")

    except Exception as e:
        _log_transition(thread_id, "analyze_profile", "scrape_fail", f"Failed to store profile data: {e}")


def get_profile_data(thread_id: str) -> dict:
    """Get profile data for a thread from profiles collection"""
    if profiles_collection is None:
        return {}

    try:
        profile_doc = profiles_collection.find_one({"thread_id": thread_id})
        thread_logger = get_thread_logger(thread_id)
        if profile_doc:
            thread_logger.info(f"Retrieved profile data for thread {thread_id}")
            return profile_doc.get("profile_data", {})
        else:
            thread_logger.info(f"No existing profile data found for thread {thread_id}")
            return {}
    except Exception as e:
        thread_logger = get_thread_logger(thread_id)
        thread_logger.error(f"Failed to retrieve profile data: {e}")
        return {}


def get_linkedin_url(thread_id: str) -> str:
    """Get linkedin_url for a thread from profiles collection"""
    if profiles_collection is None:
        return ""

    try:
        profile_doc = profiles_collection.find_one({"thread_id": thread_id})
        if profile_doc:
            return profile_doc.get("linkedin_url", "")
        return ""
    except Exception as e:
        thread_logger = get_thread_logger(thread_id)
        thread_logger.error(f"Failed to retrieve linkedin_url: {e}")
        return ""


def store_linkedin_url(thread_id: str, linkedin_url: str):
    """Stores or updates only the linkedin_url for a given thread_id."""
    thread_logger = get_thread_logger(thread_id)
    if profiles_collection is None:
        thread_logger.warning("MongoDB not available, skipping linkedin_url storage")
        return

    try:
        profiles_collection.update_one(
            {"thread_id": thread_id},
            {
                "$set": {"linkedin_url": linkedin_url, "updated_at": time.time()},
                "$setOnInsert": {
                    "created_at": time.time()
                },  # Only set created_at on first insert
            },
            upsert=True,
        )
        thread_logger.info(f"LinkedIn URL stored for thread {thread_id}")
    except Exception as e:
        thread_logger.error(f"Failed to store linkedin_url: {e}")


def get_job_description(thread_id: str) -> str:
    """Get job_description for a thread from profiles collection"""
    if profiles_collection is None:
        return ""

    try:
        profile_doc = profiles_collection.find_one({"thread_id": thread_id})
        if profile_doc:
            return profile_doc.get("job_description", "")
        return ""
    except Exception as e:
        thread_logger = get_thread_logger(thread_id)
        thread_logger.error(f"Failed to retrieve job_description: {e}")
        return ""


def store_job_description(thread_id: str, job_description: str):
    """Store job_description in profiles collection"""
    thread_logger = get_thread_logger(thread_id)
    if profiles_collection is None:
        thread_logger.warning("MongoDB not available, skipping job_description storage")
        return

    try:
        profiles_collection.update_one(
            {"thread_id": thread_id},
            {
                "$set": {"job_description": job_description, "updated_at": time.time()},
                "$setOnInsert": {"created_at": time.time()},
            },
            upsert=True,
        )
        thread_logger.info(f"Job description stored for thread {thread_id}")
    except Exception as e:
        thread_logger.error(f"Failed to store job_description: {e}")


def store_desired_job_role(thread_id: str, desired_job: str):
    """Store desired_job in profiles collection"""
    thread_logger = get_thread_logger(thread_id)
    if profiles_collection is None:
        thread_logger.warning("MongoDB not available, skipping desired_job storage")
        return

    try:
        profiles_collection.update_one(
            {"thread_id": thread_id},
            {
                "$set": {"desired_job": desired_job, "updated_at": time.time()},
                "$setOnInsert": {"created_at": time.time()},
            },
            upsert=True,
        )
        thread_logger.info(f"Job description stored for thread {thread_id}")
    except Exception as e:
        thread_logger.error(f"Failed to store job_description: {e}")


def store_conversation_message(
    thread_id: str, node: str, role: str, content: str, metadata: dict = None
):
    """Store conversation message in conversations collection"""
    thread_logger = get_thread_logger(thread_id)
    if conversations_collection is None:
        thread_logger.warning("MongoDB not available, skipping conversation storage")
        return

    try:
        conversation_entry = {
            "thread_id": thread_id,
            "node": node,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        conversations_collection.insert_one(conversation_entry)
        thread_logger.info(
            f"Conversation message stored for thread {thread_id} in node {node}"
        )

    except Exception as e:
        thread_logger.error(f"Failed to store conversation message: {e}")


def store_career_plan(
    thread_id: str, career_plan: Union[str, dict], is_review: bool = False
):
    """Store or update career plan in career_plan collection"""
    thread_logger = get_thread_logger(thread_id)
    if career_plan_collection is None:
        thread_logger.warning("MongoDB not available, skipping career plan storage")
        return

    try:
        plan_doc = {
            "thread_id": thread_id,
            "career_plan": career_plan,
            "is_review": is_review,
            "created_at": time.time() if not is_review else None,
            "updated_at": time.time(),
        }

        if is_review:
            # Update existing career plan
            existing_doc = career_plan_collection.find_one({"thread_id": thread_id})
            if existing_doc:
                plan_doc["created_at"] = existing_doc.get("created_at", time.time())

        career_plan_collection.replace_one(
            {"thread_id": thread_id}, plan_doc, upsert=True
        )

        thread_logger.info(
            f"Career plan {'updated' if is_review else 'created'} for thread {thread_id}"
        )

    except Exception as e:
        thread_logger.error(f"Failed to store career plan: {e}")


def get_career_plan(thread_id: str) -> dict:
    if career_plan_collection is None:
        return {}
    try:
        plan_doc = career_plan_collection.find_one({"thread_id": thread_id})
        thread_logger = get_thread_logger(thread_id)
        if plan_doc:
            thread_logger.info(f"Retrieved career plan for thread {thread_id}")
            return plan_doc  # full dict, not just string
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to retrieve career plan: {e}")
        return {}


# UPDATED FUNCTION FOR LINKEDIN URL CHECKING
def find_existing_profile_by_linkedin_url(linkedin_url: str, thread_id: str = None) -> dict:
    """Check if LinkedIn URL already exists and return existing profile data with thread_id"""
    thread_logger = get_thread_logger(thread_id) if thread_id else logger
    if profiles_collection is None:
        logger.warning("MongoDB not available for LinkedIn URL check")
        return {}

    try:
        existing_profile = profiles_collection.find_one({"linkedin_url": linkedin_url})
        if existing_profile:
            thread_logger.info(f"✅ EXISTING PROFILE FOUND for LinkedIn URL: {linkedin_url}")
            thread_logger.info(
                f"✅ Using existing thread_id: {existing_profile.get('thread_id')}"
            )
            thread_logger.info(f"✅ Profile data already exists - SKIPPING SCRAPING")
            return existing_profile
        else:
            thread_logger.info(
                f"❌ NO EXISTING PROFILE found for LinkedIn URL: {linkedin_url}"
            )
            thread_logger.info(f"❌ Will create NEW profile and scrape data")
            return {}
    except Exception as e:
        thread_logger.error(f"Failed to check existing profile: {e}")
        return {}


# LEGACY FUNCTIONS - KEPT FOR COMPATIBILITY
def store_thread_data(
    thread_id: str, profile_data: dict = None, conversation_entry: dict = None
):
    """Legacy function - now delegates to new storage functions"""
    if profile_data:
        store_profile_data(thread_id, profile_data)


def get_thread_data(thread_id: str) -> dict:
    """Legacy function - returns profile data for compatibility"""
    profile_data = get_profile_data(thread_id)
    return {"profile_data": profile_data} if profile_data else {}


# TOOLS
@tool
def update_desired_job_role(role) -> bool:
    """
    This updates the user's desired job role information in our database

    Args:
        role (str): the desired job role of the user in string

    Returns:
        bool : True if the information was updated successfully, False otherwise
    """
    # AgentState['desired_job_role']=role

    return True


@tool
def update_simple(simple) -> bool:
    """
    This updates the user's career plan status, if it should be simple or complex

    Args:
        simple (bool): True = simple plan, False = detailed plan

    Returns:
        bool: Returns the SAME status that was passed, so the LLM sees the correct value.
    """
    return simple


@tool
def update_analysis_status(status) -> bool:
    """
    Updates the analysis flag.
    
    Args:
        status (bool): True = analysis required, False = analysis completed
    
    Returns:
        bool: The SAME status that was passed.
    """
    return status

@tool
def update_status(status) -> bool:
    """
    Decides whether a career plan can be generated or not. Only then can the update_simple tool be called.
    
    Args:
        status (bool): True = career plan can be generated, False = career plan cannot be generated
    
    Returns:
        bool: The SAME status that was passed.
    """
    return status


@tool
def linkedin_scraper_tool(linkedin_url: str, thread_id: str) -> dict:
    """Tool to scrape LinkedIn profile data when a LinkedIn URL is detected and profile analysis is needed"""
    thread_logger = get_thread_logger(thread_id)
    thread_logger.info(f"Starting LinkedIn profile scraping for URL: {linkedin_url}")

    try:
        from scraper_utils import scrape_and_clean_profile

        api_token = os.getenv("APIFY_API_KEY")

        if not api_token:
            logger.error("APIFY_API_KEY not configured")
            raise ValueError("APIFY_API_KEY not configured")

        scraped_data = scrape_and_clean_profile(linkedin_url, api_token)

        if not scraped_data:
            logger.error("Failed to scrape profile data")
            raise ValueError("Failed to scrape profile data")

        thread_logger.info("LinkedIn profile scraped successfully")
        return scraped_data

    except Exception as e:
        logger.error(f"LinkedIn scraping failed: {e}")
        return {"error": f"Scraping failed: {e}"}


@tool
def web_search_tool(query: str, thread_id: str) -> list:
    """Tool to search the web for current market insights, industry trends, and career development resources when needed for comprehensive career planning"""
    
    thread_logger = get_thread_logger(thread_id)
    thread_logger.info(f"Starting web search for query: {query}")

    api_token = os.getenv("APIFY_API_KEY")
    if not api_token:
        logger.error("APIFY_API_KEY not configured for web search")
        return [{"error": "API token not configured"}]

    client = ApifyClient(api_token)
    # the new google search actor
    actor_id = "563JCPLOqM1kMmbbP"

    input_data = {"keyword": query, "limit": "10"}

    try:
        thread_logger.info(f"Executing web search with Apify actor {actor_id}")
        run = client.actor(actor_id).call(run_input=input_data)
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items

        results = []
        for item in dataset_items:
            # "results" is a list inside each item
            if "results" in item and isinstance(item["results"], list):
                for res in item["results"][:3]:  # only take top 3
                    result = {
                        "title": res.get("title", ""),
                        "description": res.get("description", ""),
                        "url": res.get("url", ""),
                    }
                    results.append(result)
            # stop once we collected 3 results total
            if len(results) >= 3:
                break

        thread_logger.info(f"Web search completed with {len(results)} results")
        return results[:3]

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"error": f"Web search failed: {e}"}]


# NODES


def analyze_profile_node(state: AgentState) -> Command:
    """
    Stateful entry point. Ensures profile_data is loaded. It now checks for the
    existence of the 'profile_data' key specifically to avoid the race condition.
    """
    thread_id = state["thread_id"]
    _log_transition(
        thread_id,
        "analyze_profile",
        "start",
        "Ensuring profile data is loaded into state",
    )

    if state.get("profile_data"):
        _log_transition(
            thread_id,
            "analyze_profile",
            "skip",
            "Profile data already present in state. Proceeding.",
        )
        return Command(goto="career_router")

    _log_transition(
        thread_id,
        "analyze_profile",
        "state_miss",
        "Profile data not in state. Checking database.",
    )

    # --- MODIFIED: Check for the 'profile_data' key specifically ---
    if profiles_collection is not None:
        profile_doc_from_db = profiles_collection.find_one(
            {"thread_id": thread_id, "profile_data": {"$exists": True, "$ne": {}}}
        )
    else:
        profile_doc_from_db = None

    if profile_doc_from_db:
        profile_data = profile_doc_from_db.get("profile_data", {})
        job_description = profile_doc_from_db.get("job_description")

        _log_transition(
            thread_id,
            "analyze_profile",
            "db_hit",
            f"Found complete profile in DB. Loading into state.",
        )

        return Command(
            goto="career_router",
            update={"profile_data": profile_data, "job_description": job_description},
        )

    _log_transition(
        thread_id,
        "analyze_profile",
        "db_miss",
        "No complete profile in DB. Attempting to scrape.",
    )
    linkedin_url = get_linkedin_url(thread_id)
    if not linkedin_url:
        _log_transition(
            thread_id,
            "analyze_profile",
            "interrupt",
            "Missing LinkedIn URL, cannot proceed.",
        )
        interrupt(
            {
                "task": "Welcome! To get started, please provide your LinkedIn profile URL."
            }
        )
        return END

    _log_transition(
        thread_id,
        "analyze_profile",
        "tool_call",
        f"Scraping LinkedIn URL: {linkedin_url}",
    )
    llm_with_tools = llm.bind_tools([linkedin_scraper_tool])
    scraping_prompt = f"Scrape the LinkedIn profile data from this URL: {linkedin_url} for that respective thread_id: {thread_id}"

    try:
        response = llm_with_tools.invoke(
            [HumanMessage(content=scraping_prompt, name="Scraper")]
        )
        if not response.tool_calls:
            raise ValueError("LLM failed to call the scraper tool.")

        scraped_data = linkedin_scraper_tool.invoke(response.tool_calls[0]["args"])
        if "error" in scraped_data:
            raise ValueError(f"Scraping failed: {scraped_data['error']}")

        # --- MODIFIED: Use the more robust store_profile_data function ---
        store_profile_data(thread_id, scraped_data, linkedin_url=linkedin_url)
        _log_transition(
            thread_id,
            "analyze_profile",
            "scrape_success",
            "Profile scraped. Storing and adding to state.",
        )

        limitations_prompt = f"""Based on the provided profile data, critically identify the key limitations and areas for improvement.
SCRAPED DATA:
{json.dumps(scraped_data, indent=2)}
INSTRUCTIONS:
- Be objective and specific. List the top 5-7 most significant limitations.
- Format the output as a concise, comma-separated list of phrases.
- Example: Lacks quantifiable results in experience section, 'About' section is too generic
Respond ONLY with the comma-separated list.
"""

        limitations_res = llm_call_with_retry_circuit(limitations_prompt)
        profile_limitations = limitations_res.text.strip()

        _log_transition(
            thread_id, "analyze_profile", "analysis_complete", profile_limitations
        )
        return Command(
            goto="career_router",
            update={
                "profile_data": scraped_data,
                "profile_limitations": profile_limitations,
            },
        )

    except Exception as e:
        logger.error(
            f"Profile analysis failed for thread {thread_id}: {e}", exc_info=True
        )
        error_profile = {"error": f"Failed to analyze profile: {e}"}
        return Command(
            goto="career_router",
            update={
                "messages": [
                    AIMessage(
                        "I encountered an error analyzing your profile. Let's proceed, but my context will be limited."
                    )
                ],
                "profile_data": error_profile,
            },
        )


# --- Replacement for career_router_node in agents.py ---


def career_router_node(state: AgentState) -> Command:
    """
    A hybrid router that executes multi-step plans automatically but interrupts
    to wait for user input when it's idle.
    """
    thread_id = state["thread_id"]

    # --- JOB 1: HOUSEKEEPING (Automatic) ---
    # If a task just finished, clean it up and immediately re-evaluate the state.
    if state.get("task_in_progress"):
        task_name = state["task_in_progress"]
        _log_transition(
            thread_id,
            "career_router",
            "HOUSEKEEPING",
            f"Task '{task_name}' just finished. Cleaning up. Through get task_in_progress.",
        )

        current_tasks = state.get("pending_tasks", [])
        if current_tasks and current_tasks[0] == task_name:
            current_tasks.pop(0)

        # Loop back to the router to check the new state (e.g., are there more tasks?)
        return Command(
            goto="career_router",
            update={"pending_tasks": current_tasks, "task_in_progress": None},
        )

    # --- JOB 2: EXECUTION & SYNTHESIS (Automatic) ---
    pending_tasks = state.get("pending_tasks")
    task_results = state.get("task_results")

    # If there's a plan with pending tasks, execute the next one without stopping.
    if pending_tasks:
        next_task = pending_tasks[0]
        _log_transition(
            thread_id,
            "career_router",
            "NAVIGATING",
            f"Plan in progress. Routing to next task: {next_task}",
        )
        return Command(goto=next_task, update={"task_in_progress": next_task})

    # If the plan is done and there are results, synthesize them without stopping.
    if task_results:
        _log_transition(
            thread_id,
            "career_router",
            "SYNTHESIZING",
            "Plan complete. Routing to synthesis.",
        )
        return Command(goto="synthesize_response")

    # --- JOB 3: WAITING FOR COMMAND (Interactive) ---
    # If we get here, it means: No task is running, no tasks are pending, and no results are ready.
    # The agent is completely idle. THIS is the correct and only time to interrupt.
    _log_transition(
        thread_id,
        "career_router",
        "WAITING",
        "Agent is idle. Interrupting for user command.",
    )
    resume_payload = interrupt({})  # <-- THE GRAPH PAUSES HERE
    _log_transition(
        thread_id,
        "career_router",
        "WAITING",
        f"Resumed from interrupt, processing user command. {resume_payload}",
    )

    # --- JOB 4: PLANNING (Resumes Here) ---
    # The code below ONLY executes after the API calls .invoke(Command(resume=...))
    _log_transition(
        thread_id, "career_router", "RESUMING", "Resumed with new user command."
    )

    latest_user_message = ""
    if isinstance(resume_payload, dict):
        # We will pass the user message in a dict like {"text": "the message"}
        _log_transition(
            thread_id,
            "career_router",
            "RESUMING",
            "Adding user message from dict payload.",
        )
        latest_user_message = resume_payload.get("text", "").strip()
    elif isinstance(resume_payload, str):
        _log_transition(
            thread_id,
            "career_router",
            "RESUMING",
            "Adding user message from string payload.",
        )
        latest_user_message = resume_payload.strip()

    if not latest_user_message:
        # If the resume payload was empty or malformed, loop back and wait again.
        return Command(
            goto="career_router",
            update={
                "messages": [
                    AIMessage(
                        "I received an empty command. Please tell me what you'd like to do."
                    )
                ]
            },
        )
    _log_transition(
        thread_id,
        "career_router",
        "RESUMING",
        f"User command received: {latest_user_message}",
    )
    # Store the user's new message
    store_conversation_message(thread_id, "user_input", "user", latest_user_message)

    # Get recent context for better routing decisions
    
    follow_up = state.get("follow_up", "")
    
    planner_prompt = f"""You are a hyper-intelligent task planner for a career assistant AI. Your job is to analyze a user's request and break it down into a logical, ordered sequence of executable tasks.

USER'S DESIRED JOB ROLE : {state['desired_job_role']}
USER'S CAREER PLAN: {state['career_plan']}
USER'S JOB DESCRIPTION: {state['job_description']}
USER's CAREER PLAN STATUS: {state['simple']}


LAST FOLLOW-UP QUESTION ASKED: {follow_up}

USER'S REQUEST: "{latest_user_message}"

**DELIBERATION PROCESS (Follow this internally before deciding) - VERY IMPORTANT TO FOLLOW:**
1.  **Identify the Core Verb:** What is the user's primary intent? Is it to "create," "improve," "compare," "ask," or "analyze"?
2.  **Identify the Core Noun:** What is the subject of the verb? Is it a "plan," "profile section," "job description," or a "general question"?
3.  **Match to Task:** Based on the verb-noun pair, select the single most appropriate task. If a request has multiple distinct parts, create a sequence. Example: "Analyze my fit for this JD (analyze-job) and then create a plan to bridge the gaps (create-plan)." -> ["job_fit_analysis", "career_plan"].
4. A VERY IMPORTANT NOTE(MUST ALWAYS REMEMBER): THIS IS AN IMPORTANT RULE THAT YOU NEED TO ALWAYS REMEMBER BEFORE ROUTING. IT IS ABOUT THE DISTINCTION OF THE TONE OF THE USER'S REQUEST. IF THE USER IS JUST HAVING A GENERAL CONVERSATION OR QUESTION, LIKE GREETING YOU, TELLING YOU BYE, TALKING ABOUT THE WEATHER OR ASKING ANYTHING IN THE WORLD BUT HAS NOTHING RELATED TO CAREER GUIDANCE, THEN YOU MUST COMPULSARILY ROUTE TO THE GENERAL QA NODE.

IMPORTANT NODE ROUTING INSTRUCTION:
- If user's desired job role is not present and user's career plan is missing and they have any questions related to their career and what they should do and how they should achieve their goals, then you should COMPULSARILY route to the career_plan node
- User desired job role is NOT PRESENT, ALWAYS GO TO CAREER PLAN NODE ONLY. IF YOU SEE AN INPUT LIKE : "do a job fit analysis: ", then YOU MUST COMPULSARY GO TO CAREER PLAN NODE.

**CRITICAL ROUTING RULE: UNDERSTAND THE USER'S REFERENCE (IF APPLICABLE)**: 
1. This is a very critical routing rule. Understand from the USER'S REQUEST if the user is making a reference to any of the previous response given. This is VERY IMPORTANT to understand, since this impacts the routing as well. If you understand that the user is refering to something generated in some of the previous previous messages, then you MUST COMPULSARILY route by taking that AI RESPONSE and USER MESSAGE FROM into consideration.
2. Always understand what the user is asking for by understanding the latest user message.
3. If the user is making a reference to any of the previous responses given, (understand that from the latest user message), then you MUST COMPULSARILY take that AI RESPONSE AND USER MESSAGE and LAST FOLLOW-UP QUESTION ASKED FROM {follow_up} into consideration and must use that to answer what the user is precisely asking for.

**CRITICAL FOLLOW-UP QUESTION RULE (ONLY APPLICABLE FOR FOLLOW-UP QUESTIONS)**:
- If the user is responding to a follow-up question, then you MUST COMPULSARILY take the LAST FOLLOW-UP QUESTION ASKED FROM {follow_up}, PREVIOUS USER MESSAGE AND AI RESPONSE refering to that follow up into consideration and must use that to answer what the user is precisely asking for. This routing situation must be done in context of these three variables, very strictly.
- If a very detailed follow-up question is asked, pointing to only a specific section or task of the previous response, always you MUST COMPULSARILY route to general_qa node.

UNDERSTANDING CAREER PLAN STATUS:
- If the user's career plan status is 'true', and the user is asking for a detailed career plan, you should COMPULSARILY route to the CAREER PLAN NODE.

AVAILABLE TASKS (NODES):
1. job_fit_analysis - Route here ONLY when:
   - Use only when USER'S JOB DESCRIPTION is present
   - User provides or mentions a specific job description/posting they want to be analyzed against
   - User explicitly asks about their fit/match for a particular role
   - User wants to compare their profile against job requirements
   - User asks about "job fit", "job match", "am I suitable for this role"
   - Keywords: job description, job posting, fit analysis, job match, role suitability, application review

2. enhance_profile - Route here ONLY when:
   - Use when the user explicitly asks to improve, rewrite, or optimize any section of their profile.
   - User wants to improve/rewrite specific LinkedIn profile sections (About, Experience, Skills, etc.)
   - User asks for profile optimization or enhancement
   - User wants help with profile content, wording, or structure
   - User mentions improving headlines, summaries, or any profile section
   - Keywords: improve profile, rewrite about section, optimize LinkedIn, enhance profile, fix my profile, profile content

3. career_plan - Route here ONLY when:
   - Use ONLY when the user asks for a time-based, step-by-step plan or roadmap to achieve a specific career goal.
   - User explicitly requests a career plan, roadmap, or career development strategy
   - User asks about career transition planning or next career steps
   - User wants skill development roadmap or career growth planning
   - User asks to create/modify/review their career plan or ANY OTHER CHANGES TO THE CAREER PLAN. 
   - Keywords: career plan, career roadmap, development plan, career transition, growth strategy, career goals



4. general_qa - Route here ONLY when:
   - ANY GENERAL QUESTION IN THE WORLD, THAT MAY OR MAY NOT BE RELATED TO THE USER'S PROFILE OR CAREER GOAL. 
   - Use for all other general career questions, advice, or when the user asks for a list of options (e.g., "what jobs can I apply for?").
   - User asks general career advice or questions (interview tips, networking, salary negotiation)
   - User has profile-related questions but not requesting profile enhancement
   - User asks about resume tips, cover letters, or general career guidance
   - User asks miscellaneous career-related questions that don't fit other categories
   - User asks about their current profile/background but not requesting changes
   - Keywords: career advice, interview tips, general questions, how to, career guidance, profile questions, any future related questions


5. user_interaction - Route here ONLY as absolute last resort when (MUST BE THE LAST RESORT):
   - User's request is completely unclear or ambiguous
   - Request doesn't fit any above categories at all
   - Use this sparingly - most requests should fit in general_qa if unclear

CLASSIFICATION INSTRUCTIONS:
- Read the user's request carefully and identify the PRIMARY intent, ALWAYS TAKE THE PREVIOUS AI RESPONSE and PREVIOUS USER MESSAGE into consideration.
- Be very specific about which category matches best
- Only use user_interaction if truly cannot classify


CRITICAL RULES FOR PLANNING:
1.  Decomposition: Break down complex requests into multiple, sequential tasks. The order in the list is the order of execution.

2.  Strict Classification: Be precise.
    -   "give me a 6-month plan to become a data analyst" -> ["career_plan"]
    -   "what are some future jobs i can aim for?" -> ["general_qa"] (This is a request for a LIST, NOT a plan).
    -   "Check my fit for this JD and then give me a plan to get the job." -> ["job_fit_analysis", "career_plan"]

Respond with ONLY a valid JSON list of task strings. Do not include any explanation.
"""
    try:
        response = llm_call_with_retry_circuit(planner_prompt)
        cleaned_response = (
            response.text.strip().replace("`json", "").replace("`", "").strip()
        )
        tasks = json.loads(cleaned_response)
        _log_transition(
            thread_id, "career_router", "PLANNING_SUCCESS", f"Plan created: {tasks}"
        )

        # We have a new plan. Update the state and loop back to the router.
        # On the next run, JOB 2 will see the `pending_tasks` and start executing automatically.
        return Command(
            goto="career_router",
            update={
                "messages": [
                    HumanMessage(content=latest_user_message, name="CareerRouter")
                ],
                "pending_tasks": tasks,
                "task_results": {},  # Reset results for the new plan
            },
        )
    except Exception as e:
        logger.error(f"Planner failed: {e}. Defaulting to user_interaction.")
        return Command(
            goto="career_router",
            update={
                "messages": [
                    HumanMessage(
                        content=latest_user_message, name="CareerRouterException"
                    )
                ],
                "pending_tasks": ["user_interaction"],
                "task_results": {},
            },
        )


# This node's logic is correct. NO CHANGES.
def synthesize_response_node(state: AgentState) -> Command:
    thread_id = state["thread_id"]
    task_results = state.get("task_results", {})
    thread_logger = get_thread_logger(thread_id)

    latest_user_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user_message = msg.content
            break

    if task_results:
        thread_logger.info(f"type = {type(task_results)}")
        
        # Get context for synthesis
        # recent_ai_response = state.get("recent_ai_response", "")
        # previous_user_message = state.get("previous_user_message", "")
        current_follow_up = state.get("follow_up", "")
        
        synthesis_prompt = f"""You are a helpful AI assistant, WHOSE ONLY JOB IS TO PRINT AND COMBINE THE RESULTS OF INTERNAL TASKS, WITH NO CHANGES TO BE MADE, and conversational orchestrator. Your job is to combine the results of several internal tasks into a single response that directly answers the user's original question, AND generate exactly one follow-up question to maintain conversational flow.

USER'S ORIGINAL QUESTION: "{latest_user_message}"
LAST FOLLOW-UP QUESTION ASKED: {current_follow_up}
CAREER PLAN STATUS: {state['simple']}
ANALYSIS STATUS: {state['analysis']}
CAREER PLAN GENERATION STATUS: {state['status']}
USER's DESIRED JOB ROLE: {state['desired_job_role']}

INTERNAL TASK RESULTS:
{json.dumps(task_results, indent=2)}

FORMAT:
YOU MUST STRICTLY FOLLOW THE FINAL OUTPUT SCHEMA. AND EXACTLY FOLLOW THE RULES MENTIONED IN THE SCHEMA.

MOST IMPORTANT RULE: THE OUTPUT MUST NEVER BE RETURNED IN JSON FORMAT. IT SHOULD ALWAYS BE A WELL FORMATTED RESPONSE

Generate the complete response now."""


        messages = [{"role":"assistant", "content": synthesis_prompt}]+state['messages']
        llm_with_schema = llm.with_structured_output(FinalOutput)
        structured_response = llm_with_schema.invoke(
            messages
        )
        result=structured_response.model_dump()
        
        # final_response = result['output']
        follow_up_question = result['follow_up']
        # response = llm_call_with_retry_circuit(synthesis_prompt).text
        final_output=f"{result['intro']}\n\n" + "\n".join(result['output']) + f"\n\n{result['follow_up']}"
        # _log_transition(thread_id, "synthesis", "final_output", f"Final output: {final_output}")
        
        # # Extract follow-up question (assume it's the last line/paragraph)
        # response_parts = response.strip().split('\n\n')
        # if len(response_parts) >= 2:
        #     main_response = '\n\n'.join(response_parts[:-1])
        #     follow_up_question = 
        # else:
        #     main_response = response.strip()
        #     follow_up_question = "Would you like me to explore this further or finalize your plan?"
        
        # # Combine main response with follow-up question
        # final_response = f"{main_response}\n\n{follow_up_question}"
        
        store_conversation_message(thread_id, "synthesis", "assistant", final_output)

        return Command(
            goto="career_router",
            update={
                "messages": [AIMessage(content=final_output, name="Synthesis")],
                "task_results": None,
                "pending_tasks": None,
                
                "follow_up": follow_up_question
            },
        )

    else:
        final_response = "Goodbye! Let me know if you need anything else."

    # store_conversation_message(thread_id, "synthesis", "assistant", final_response)

    return Command(goto="career_router")


# --- Replacement for job_fit_analysis_node ---

# --- REPLACE the entire enhance_profile_node in agents.py ---


def enhance_profile_node(state: AgentState) -> Command:
    """
    Enhances the user's profile, tailoring suggestions towards the target job description in the state.
    """
    thread_id = state["thread_id"]
    _log_transition(
        thread_id, "enhance_profile", "start", "Starting goal-aware profile enhancement"
    )

    profile_data = state.get("profile_data")
    profile_limitations = state.get("profile_limitations")
    # --- NEW: Fetch the target job description from state ---
    job_description = state.get(
        "job_description", "No specific job goal has been set yet."
    )

    if not profile_data or "error" in profile_data:
        current_results = state.get("task_results", {})
        new_results_on_error = {
            **current_results,
            "enhance_profile": "Task skipped: Profile data is not loaded.",
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

    latest_user_request = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user_request = msg.content.strip()
            break

    profile_text = json.dumps(profile_data, indent=2)

    # Get recent context for understanding references
    # recent_ai_response = state.get("recent_ai_response", "")
    follow_up = state.get("follow_up", "")
    # logger.info(f"Recent AI response from enhance_profile_node: {recent_ai_response}")
    
    enhancement_prompt = f"""You are a world-class LinkedIn profile writer and career coach. Your task is to enhance the user's profile based on their specific request, with a strong focus on aligning it with their target job.

**CONTEXT:**
1.  **USER REQUEST:** "{latest_user_request}"
2.  **TARGET JOB DESCRIPTION / GOAL:** 
    {job_description}
3. **USER'S DESIRED JOB ROLE**: {state.get('desired_job_role')}

4.  **USER'S FULL PROFILE:**
    {profile_text}
5.  **PREVIOUSLY IDENTIFIED PROFILE LIMITATIONS:**
    {profile_limitations if profile_limitations else "None identified."}



6.  **LAST FOLLOW-UP QUESTION ASKED:** 
    {follow_up}

**VERY IMPORTANT AND CRITICAL RULE TO FOLLOW (MUST FOLLOW)**:
- ANY AND ALL OF THE OUTPUT YOU PROVIDE MUST BE STRICTLY WITHIN THE WORD LIMIT OF 210-230 WORDS ONLY. THIS MUST NOT BE EXCEEDED IN ANY SITUATION. THIS RULE MUST BE FOLLOWED ALL THE TIME. IT IS A STRICT AND CRITICAL BOUNDARY TO BE MAINTAINED. YOU MUST ALWAYS BE VERY PRCISE, CONCISE AND TO THE POINT, NO BEATING AROUND THE BUSH.

UNDERSTANDING USER'S REFERENCE:
- Always understand what the user is asking for by understanding the latest user message.
- If the user is making a reference to any of the previous responses given, (understand that from the latest user message), then you MUST COMPULSARILY take the PREVIOUS AI RESPONSES into consideration and must use that to answer what the user is precisely asking for.
- Consider the LAST FOLLOW-UP QUESTION to understand if the user is responding to that specific question.

**CRITICAL DIRECTIVE: ENHANCE FOR THE TARGET JOB**
Your primary goal is to rewrite and enhance the requested profile section to make the user a more compelling candidate for their **DESIRED JOB ROLE**.

1.  **Identify Target Section:** Determine which section of the profile the user wants to enhance.
2.  **Analyze Keywords:** Extract key skills, technologies, and qualifications from the **TARGET JOB DESCRIPTION** (If Applicable), or else analyze the user's COMPLETE profile in detail and to the point what the user is asking.
3.  **Weave in Keywords:** When rewriting the section, you MUST strategically incorporate these keywords and concepts, while maintaining factual integrity.
4.  **Explain Every Change:** You MUST include a "Rationale for Changes" section. Justify each change by explaining how it better aligns the user's profile with the target job's requirements.

NO GENERIC ADVICE MUST BE PROVIDED, IT MUST ALWAYS BE TO THE POINT, CONCISE AND ANSWER ONLY TO THE USER'S QUESTION.
Provide ONLY the enhanced content and the detailed explanation while strictly adhering to the word limit mentioned.
"""

    try:
        #changing the code to give the invoke(messages) thing
        messages =[{"role":"assistant", "content": enhancement_prompt}]+state['messages']
        initial_response=llm.invoke(messages)
        
        
        result = initial_response.content
        current_results = state.get("task_results", {})
        new_results = {**current_results, "enhance_profile": result}
        return Command(goto="career_router", update={"task_results": new_results})
    except Exception as e:
        logger.error(
            f"Profile enhancement failed for thread {thread_id}: {e}", exc_info=True
        )
        current_results = state.get("task_results", {})
        error_message = f"Error: Profile enhancement could not be completed. {e}"
        new_results_on_error = {
            **current_results,
            "enhance_profile": error_message,
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )


# --- REPLACE the entire job_fit_analysis_node in agents.py ---

# --- REPLACE the entire job_fit_analysis_node in agents.py with this code ---


def job_fit_analysis_node(state: AgentState) -> Command:
    """
    Performs job fit analysis strictly using the provided job description (no web search).
    Enforces the JobFitAnalysis structured output schema for consistency.
    """
    thread_id = state["thread_id"]
    thread_logger = get_thread_logger(thread_id)
    _log_transition(
        thread_id, "job_fit_analysis", "start", "Starting goal-aware job fit analysis"
    )

    profile_data = state.get("profile_data")
    if not profile_data or "error" in profile_data:
        _log_transition(
            thread_id,
            "job_fit_analysis",
            "fail",
            "Profile data not available in state.",
        )
        current_results = state.get("task_results", {})
        new_results_on_error = {
            **current_results,
            "job_fit_analysis": "Task failed: I cannot perform a job fit analysis without your profile data.",
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

    # Get the most recent user message
    latest_user_message = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user_message = msg.content.strip()
            break

    # Use the latest user message as the job description
    current_job_description = latest_user_message
    if not current_job_description:
        error_message = "No job description provided. Please paste a JD for analysis."
        logger.error(error_message)
        return Command(
            goto="career_router",
            update={"task_results": {"job_fit_analysis": error_message}},
        )

    _log_transition(
        thread_id,
        "job_fit_analysis",
        "goal_storage",
        "Storing new target job description.",
    )
    store_job_description(thread_id, current_job_description)

    profile_text = json.dumps(profile_data, indent=2)

    # Get recent context for understanding references
    # recent_ai_response = state.get("recent_ai_response", "")
    follow_up = state.get("follow_up", "")
    # logger.info(
    #     f"Recent AI response from job_fit_analysis_node: {recent_ai_response}"
    # )

    thread_logger.info(f"job description: {current_job_description}")

    # Prompt for structured Job Fit Analysis
    job_fit_prompt = f"""You are a senior technical recruiter. Perform a rigorous job fit analysis by comparing the candidate's profile against the job description.

JOB DESCRIPTION:
{current_job_description[:4000]}

**USER'S DESIRED JOB ROLE**: {state.get('desired_job_role')}


**LAST FOLLOW-UP QUESTION ASKED:** 
{follow_up}

CANDIDATE'S FULL PROFILE:
{profile_text}

**CRITICAL ANALYSIS DIRECTIVE:**
- Go beyond keyword matching. 
- For each strength, gap, or recommendation, provide a justification explicitly tied to the candidate's profile.
- Be concise, but clear enough for a mobile user.
- Output must follow the JobFitAnalysis structured schema exactly.
"""

    try:
        #the invoke(messages) change
        messages = [{"role":"assistant", "content": job_fit_prompt}]+state['messages']
        # initial_response=llm.invoke(messages)
        
        # result = initial_response.content
        
        llm_with_schema = llm.with_structured_output(JobFitAnalysis)
        structured_job_fit = llm_with_schema.invoke(
            messages
        )
        result=structured_job_fit.model_dump()
        thread_logger.info(f"Job fit analysis result: {structured_job_fit}")
        
        # logger.info(f"Job fit analysis JSON: {result}")

        current_results = state.get("task_results", {})
        new_results = {**current_results, "job_fit_analysis": result}

        return Command(
            goto="career_router",
            update={
                "task_results": new_results,
                "job_description": current_job_description,  # Update state with the new goal
            },
        )

    except Exception as e:
        logger.error(
            f"Job fit analysis failed for thread {thread_id}: {e}", exc_info=True
        )
        current_results = state.get("task_results", {})
        error_message = f"I'm sorry, but I encountered an error while performing the job fit analysis: {e}. Please try rephrasing your request."
        new_results_on_error = {**current_results, "job_fit_analysis": error_message}
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )



# --- REPLACE the entire career_plan_node in agents.py ---

# --- REPLACE the entire career_plan_node function with this new, more forceful version ---


def career_plan_node(state: AgentState) -> Command:
    """
    Creates a career plan. This version is radically simplified to force the LLM
    to use the provided profile data and the tools included like update_desired_job_role, update_simple, update_analysis_status and web_search_tool without deviation.
    """
    thread_id = state["thread_id"]
    _log_transition(
        thread_id,
        "career_plan",
        "start",
        "Starting forceful, goal-aware career planning",
    )
    profile_data = state.get("profile_data")
    if not profile_data or "error" in profile_data:
        _log_transition(
            thread_id, "career_plan", "fail", "Profile data not available in state."
        )
        current_results = state.get("task_results", {})
        new_results_on_error = {
            **current_results,
            "career_plan": "Task failed: I cannot create a personalized plan without your profile data.",
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

    latest_user_request = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            latest_user_request = msg.content.strip()
            break

    profile_text = json.dumps(profile_data, indent=2)

    #temp variables
    temp_role = state.get("desired_job_role")
    temp_analysis = state.get("analysis")
    temp_state = state.get("simple")
    temp_status = state.get("status")

    # Get recent context for understanding references
    follow_up = state.get("follow_up", "")

    _log_transition(
        thread_id, "career_plan", "statuses", f"All status before LLM: simple:{state['simple']}, analysis:{state['analysis']}, status:{state['status']}"
    )
    
    # --- THIS PROMPT IS RADICALLY SIMPLIFIED AND MORE DIRECTIVE ---
    unified_prompt_1a = f"""You are an elite, holistic AI career strategist. Your task is to create a hyper-personalized and actionable career plan.

**CONTEXT:**
1.  **USER'S LATEST REQUEST:** "{latest_user_request}"
2. **DESIRED JOB ROLE** :{state['desired_job_role']}
3. **USER'S CAREER PLAN STATUS (IF SIMPLE/COMPLEX):** {state['simple']}
4. **USER'S ANALYSIS STATUS:** {state['analysis']}
5. **USER'S STATUS (IF CAREER PLAN CAN BE GENERATED OR NOT):** {state['status']}

6.  **USER'S FULL PROFILE DATA:**
    {profile_text}

7.  **LAST FOLLOW-UP QUESTION ASKED:** 
    {follow_up}

"""

    unified_prompt_1b = f"""
You are an elite, holistic AI career strategist. Your task is to create a hyper-personalized and actionable career plan.

**CONTEXT:**
1.  **USER'S LATEST REQUEST:** "{latest_user_request}"
2. **DESIRED JOB ROLE** :{state['desired_job_role']}
3. **USER'S CAREER PLAN STATUS (IF SIMPLE/COMPLEX):** {state['simple']}
4. **USER'S ANALYSIS STATUS:** {state['analysis']}
5. **USER'S STATUS (IF CAREER PLAN CAN BE GENERATED OR NOT):** {state['status']}

6.  **USER'S FULL PROFILE DATA:**
    {profile_text}

7.  **LAST FOLLOW-UP QUESTION ASKED:** 
    {follow_up}
 8. **USER'S TIMELINE FOR CAREER PLAN:**: {state['career_plan'][0]['duration'] if state.get('career_plan') and isinstance(state['career_plan'], list) and len(state['career_plan']) > 0 and isinstance(state['career_plan'][0], dict) and 'duration' in state['career_plan'][0] else 'Not specified'}

"""

    unified_prompt_2 = """

UNDERSTANDING USER'S REFERENCE:
- Always understand what the user is asking for by understanding the latest user message.
- If the user is making a reference to the previous response given, (understand that from the latest user message), then you MUST COMPULSARILY take the RECENT AI RESPONSES into consideration and must use that to answer what the user is precisely asking for.
- Consider the LAST FOLLOW-UP QUESTION to understand if the user is responding to that specific question.

*STEP 1: DELIBERATION ON USER'S NEW INTENT (CRITICAL!)*
Before generating a plan, the user's desired job role must be present, if not you must ask the user what their desired job role or career goal is. 
- call the update_desired_job_role tool for updating the role if the same job role already exists in the data then do not call this tool.
- If the desired job role is not present, then determine the role basis the conversation and call the update_desired_job_role tool.

- CONDITIONS TO UNDERSTAND FOR CALLING THE update_status tool, update_simple tool and update_simple to update the analysis status and career plan status:
    You are controlling three tools to manage the flow of job analysis and career planning: 
    - update_analysis_status (boolean)
    - update_status (career_plan_status, boolean)
    - update_simple (boolean for simple vs detailed)

    You must ALWAYS decide which of these tools to call based on the user's latest request, according to the following strict rules. IMPORTANT NOTE: THE TOOL CALLS MUST BE VERY STRICTLY TAKEN TO ANSWER ONLY WHAT THE USER IS ASKING:

    1. JOB ROLE CHANGE:
    - Whenever the update_desired_job_role tool is called, you MUST BE VERY MINDFUL OF THESE RULES:
        - FIRST UNDERSTAND WHAT THE USER IS ASKING FOR WHEN THEY ARE CHANGING THE JOB ROLE. BASIS THAT UNDERSTANDING, CALL THE CORRECT TOOLS. WHILE CHANGING THE JOB ROLE, THE USER CAN ASK FOR A DETAILED/SIMPLE CAREER PLAN OR JUST CHANGE JOB ROLE.
        - (IMPORTANT RULE - ONLY IF THE USER IS ONLY CHANGING THE JOB ROLE AND NOT ASKING FOR A CAREER PLAN); Call update_analysis_status → True AND Call update_status → False (career plan disabled until analysis finishes).
        - IMPORTANT RULE: BUT IF THE USER ASKS FOR A CAREER PLAN (ANY TYPE), then you must call the update_status tool to update the career plan status to bool value True and depending on the user's request, the update_simple tool must be called to update the simple status to bool value True or False for a simple/detailed plan respectively.
        - IF THE USER IS CHANGING THE JOB ROLE AND ASKING FOR A CAREER PLAN, THEN YOU MUST CALL THE update_status tool to update the career plan status to bool value True and depending on the user's request, the update_simple tool must be called to update the simple status to bool value True or False for a simple/detailed plan respectively. YOU MUST NOT CALL update_analysis_status tool in this case (SINCE THE USER IS SPECIFICALLY ASKING FOR A CAREER PLAN).
        - THIS MUST BE PURELY DETERMINED BASED ON THE USER'S REQUEST AND MUST BE VERY STRICTLY FOLLOWED.

    2. EXPLICIT ANALYSIS REQUEST:
    - If the user explicitly asks for an analysis (short analysis, profile analysis, role fit analysis, etc.):
        - Call update_analysis_status → True.
    - After analysis is provided, you MUST call update_analysis_status → False.

    3. EXPLICIT CAREER PLAN REQUEST - VERY IMPORTANT RULE:
    - If the user explicitly asks for a career plan (any type):
        - IF THE USER DOESN'T MENTION THE TYPE OF CAREER PLAN, THEN YOU MUST ASSUME THAT THE USER IS ASKING FOR A SIMPLE CAREER PLAN.
        - Call update_status → True.
        - update_analysis_status → False.
        - THEN call update_simple:
        - update_simple → True if the user asks for a simple plan.
        - update_simple → False if the user asks for a detailed plan. MAKE SURE TO CALL web_search_tool as well while generating DETAILED CAREER PLAN ONLY.

    4. COMBINATIONS:
    - User may ask for both analysis + plan. In that case, call the tools together:
        - update_analysis_status (as per rules above).
        - update_status → True (career plan requested).
        - update_simple → True/False depending on simple or detailed request.

    5. STRICT FLOW ENFORCEMENT:
    - NEVER call update_simple before update_status.
    - If user is NOT asking for any career plan, call update_status → False.
    - For EVERY user request, you must return the correct tool calls — no exceptions.

    The system depends on these flags, so you MUST apply these rules precisely. Do not invent extra logic or outputs.


Then understand the user's LATEST request. The "Existing Career Plan" is provided for historical context ONLY.
- call the update_simple tool to update the career plan status to bool value False if the user is asking for a detailed career plan. 
- If the user is explicitely asking for a detailed career plan or if the user's reply intends that they are AGREEING to the question asked at the end of every simple career plan i.e, (Do you need me to make a more detailed career plan for you?), or they respond in a way which insinuates that they need a detailed plan, then you must compulsarily call the update_simple tool to update the career plan status to bool value False.
- If there is a change in the job role in the user's profile, i.e, when the update_desired_job_role tool has been called, it means the user has changed their job role and is looking a short analysis for that role, so you must compulsarily call the update_analysis_status tool to update the analysis status to bool value True.

-   *Analyze the LATEST REQUEST:* Is the user asking for a completely new plan, a modification of the timeline, or a different goal?
-   *DO NOT BLINDLY COPY:* You must not copy sections from the "Existing Career Plan" (like a "CV vs. UI/UX" analysis) if they are NOT relevant to the user's *current, specific request. Your new plan must be a direct answer to their latest prompt. A request to "make that plan for 10 months now" implies creating a new, detailed 10-month version of the *career plan itself, not repeating previous analytical tasks.

---
*STEP 2: THE HOLISTIC PERSONALIZATION MANDATE*

IF THE USER'S ANALYSIS STATUS IS "True", then the below steps are MANDATORY to follow:
MANDATE FOR ANALYSIS:
**MOST IMPORTANT RULE: THE ENTIRE ANALYSIS GENERATED MUST BE STRICTLY BETWEEN 60-65 WORDS ONLY**
Your response will be considered a *FAILURE* unless you follow these rules precisely.
1. ANALYZE THE PROFILE FOR KEYWORDS:
    - Quickly scan the user's profile for their primary skills, most recent projects, and certifications. You will use these as a foundation.
    - Do not perform a deep, exhaustive analysis. The goal is a quick, high-level summary.
STRICTLY FOLLOW THE ANALYSIS SCHEMA TO GENERATE THE ANALYSIS.
------
IF USER'S CAREER PLAN STATUS IS "True", then the below steps are MANDATORY to follow:
MANDATE FOR SIMPLE CAREER PLAN:
**MOST IMPORTANT RULE: THE ENTIRE SIMPLE PLAN GENERATED MUST BE STRICTLY BETWEEN 140-160 WORDS ONLY**
Your response will be considered a *FAILURE* unless you follow these rules precisely.
1. ANALYZE THE PROFILE FOR KEYWORDS:
    - Quickly scan the user's profile for their primary skills, most recent projects, and certifications. You will use these as a foundation.
    - Do not perform a deep, exhaustive analysis. The goal is a quick, high-level summary.
2. POPULATE THE SIMPLE PLAN (Adhere Strictly to the Schema):
    - Strictly follow the SimpleCareerPlan schema to generate the simple plan.
3. NO DEEP EXPLANATIONS:
You MUST NOT provide a "Personalization Rationale" or any other detailed justification for the simple plan. The output should be a clean, straightforward list of items corresponding to the schema fields. Your entire output must conform to the SimpleCareerPlan structured output format.

ALWAYS, after generating the simple plan, ALWAYS END THE PLAN WITH: Do you need a more detailed plan?

------
IF USER'S CAREER PLAN STATUS IS "False", then the below steps are MANDATORY to follow:
MANDATE FOR DETAILED CAREER PLAN:
Your response will be considered a *FAILURE* unless you follow these rules precisely. Generic advice is forbidden.
**MOST IMPORTANT RULE: THE ENTIRE DETAILED PLAN GENERATED MUST BE STRICTLY MUST BE BETWEEN 2000-2200 CHARACTERS ONLY**

Strictly follow the CareerPlan structured output format.

YOU CANNOT EXCEED THE MENTIONED CHARACTER LIMITS. IF YOU DO, YOUR RESPONSE WILL BE CONSIDERED A FAILURE.
------
CRITICAL DECISION FRAMEWORK FOR WEB SEARCH:

Your another important task is to analyze the user's request to determine if a web search for current market data is necessary. Follow these rules strictly. Only return the web search results, do not generate random URLs on your own.

- A web search is MANDATORY (**COMPULSARY ONLY IF USER'S CAREER PLAN STATUS IS "False"**) if the request involves:
  1.  A Brand New Plan: The user has no existing career plan.
  2.  A Fundamental Change or Overhaul: Even if a plan exists, a search is required if the user wants to change:
      - The target job title or career path (e.g., "Change my plan to become a marketing manager," "I want to switch to data science now").
      - The core industry (e.g., "How can I apply my skills in the healthcare industry instead?").
      - A drastic change in timeline or intensity that implies a new strategy (e.g., "I need an intensive 3-month plan to get job-ready").
  3.  A Request for Current Market Data: The user explicitly asks for the latest trends, in-demand skills, up-to-date salary expectations, or emerging technologies in their field.

- A web search is NOT NECESSARY if the user is asking for:
  - Minor Tweaks or Modifications: The user wants to adjust parts of an existing plan without changing the core career goal (e.g., "Can you adjust the timeline from 6 to 8 months?", "Suggest a different book for skill X").
  - Clarification or Rephrasing: The user wants to understand a part of their current plan better.

- WHAT SHOULD BE THE WEB SEARCH QUESTION:
    - This must be very specific to the user's request, the user's desired job role and their profile data.
    - This search question is dynamic and must not be generic. And must provide results that can actually help the user in their career plan.
    - YOU MUST NOT OVERLOAD YOUR WEB SEARCH QUERY WITH TOO MANY KEYWORDS. KEEP IT SHORT, CONCISE AND TO THE POINT.
-----
*OUTPUT STRUCTURE:*
Provide a detailed, step-by-step plan based on the user's requested timeline. Use clear headings. For each step, include a "Personalization Rationale" subsection that explains the connection to their profile.

----
CAREER PLAN STRUCTURE:
- this is handled in the 'CareerPlan' and 'SimpleCareerPlan' structured output class, based on career plan status structured output class, follow ONLY that in detail.
----
CAREER PLAN REVIEW:
- Only the necessary parts of my plan must be changed/modified as mentioned by the user in their prompt. Be very specific about the changes that are to be made. 
- The changes mentioned by the user ONLY must be made. No other changes must be made. These changes can be in terms of skills, timeline, etc.
- IMPORTANT RULE: IF THE USER IS ASKING FOR A CHANGE, ADHERE TO WHAT THE USER IS ASKING FOR ONLY. DO NOT MAKE CHANGES THAT ARE NOT MENTIONED BY THE USER. STRICTLY FOLLOW THE USER'S REQUEST.

GUIDELINES:
- Be highly specific and actionable
- Tailor to user's exact request and background. No sort of extra information or advice must be given. Only the plan as asked by the user pertaining to the user's profile only, personlized.
- Use web search when additional insights would strengthen the plan
- If modifying existing plan, clearly integrate requested changes
- Provide realistic timelines and achievable goals

Begin now. Decide if a web search is needed, then generate the deeply personalized plan.
"""


    # if state['career_plan'][0]['duration'] is None:
    #     unified_prompt=unified_prompt_1a+unified_prompt_2
    # else:
    #     unified_prompt=unified_prompt_1a+unified_prompt_1b+unified_prompt_2
    
    if (
        state.get("career_plan")
        and isinstance(state["career_plan"], list)
        and len(state["career_plan"]) > 0
        and state["career_plan"][0].get("duration")
    ):
        unified_prompt = unified_prompt_1a + unified_prompt_1b + unified_prompt_2
    else:
        unified_prompt = unified_prompt_1a + unified_prompt_2


    
    messages = [{"role": "system", "content": unified_prompt}] + state["messages"]
    llm_with_tools = llm.bind_tools(
        [web_search_tool, update_desired_job_role, update_simple, update_analysis_status, update_status]
    )


    initial_response = llm_with_tools.invoke(messages)

    final_messages = [
        HumanMessage(content=unified_prompt, name="CareerPlanProcedural"),
        initial_response,
    ]

    
    _log_transition(
        thread_id, "career_plan", "statuses", f"All status after LLM after temp variables: simple:{state['simple']}, analysis:{state['analysis']}, status:{state['status']}"
    )

    _log_transition(thread_id, "career_plan", "tool_call", f"All the tool calls: {initial_response.tool_calls}")
    # tool thing'
    if_web_search = False
    if len(initial_response.tool_calls) > 0:
        _log_transition(thread_id, "career_plan", "tool_call", "LLM is using tools.")
        for tool_call in initial_response.tool_calls:
            if tool_call["name"] == "web_search_tool":
                if_web_search = True
                tool_result = web_search_tool.invoke(tool_call["args"])
                final_messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result), tool_call_id=tool_call["id"]
                    )
                )

            elif tool_call["name"] == "update_desired_job_role":
                #Args: role (str): the desired job role of the user in string format
                temp_role = tool_call["args"]["role"]
                update_desired_job_role.invoke(tool_call["args"])
                store_desired_job_role(thread_id, temp_role)
                # temp_state = True

            elif tool_call["name"] == "update_simple":
                new_status = tool_call["args"]["simple"]  # Extract the boolean value
                update_simple.invoke(tool_call["args"])
                _log_transition(thread_id, "career_plan", "statuses", f"Career plan status updated ={new_status}")
                temp_state = new_status

            elif tool_call["name"] == "update_analysis_status":
                temp_analysis = tool_call["args"]["status"]
                update_analysis_status.invoke(tool_call["args"])
                _log_transition(thread_id, "career_plan", "statuses", f"Analysis status updated ={temp_analysis}")
            
            elif tool_call["name"] == "update_status":
                temp_status = tool_call["args"]["status"]
                update_status.invoke(tool_call["args"])
                _log_transition(thread_id, "career_plan", "statuses", f"Status updated ={temp_status}")


    _log_transition(thread_id, "career_plan", "statuses", f"Analysis status ={temp_analysis}, Career plan blocker status ={temp_status}, Simple status ={temp_state}")
    #analysis and career plan schema stuff now combined also works
    structured_outputs = []

    if temp_analysis == True:
        _log_transition(thread_id, "career_plan", "schema_choice", "Generating ANALYSIS")
        structured_outputs.append(llm_with_tools.with_structured_output(Analysis))

    if temp_status == True:
        _log_transition(thread_id, "career_plan", "statuses", f"Final plan status before structured output: {temp_status} ({type(temp_status)})")
        if temp_state == True:
            _log_transition(thread_id, "career_plan", "schema_choice", "Generating SIMPLE career plan.")
            structured_outputs.append(llm_with_tools.with_structured_output(SimpleCareerPlan))
        else:
            _log_transition(thread_id, "career_plan", "schema_choice", "Generating DETAILED career plan.")
            structured_outputs.append(llm_with_tools.with_structured_output(CareerPlan))

    else:
        _log_transition(thread_id, "career_plan", "statuses", f"ntg happened for the career plan part: analysis={temp_analysis}, status={temp_status}")
        
    if if_web_search == True:
        final_prompt = "Based on all available information and the provided search results, generate the complete, hyper-personalized career development plan. **CRITICAL REMINDER:** You MUST use the provided profile data and you MUST format the web search results under a `## Sources from the Web` heading with real, clickable Markdown links."
        final_messages.append(
            HumanMessage(content=final_prompt, name="CareerPlanProceduralFinal")
        )

    try:
        results = []
        for x in structured_outputs:
            temp = x.invoke(final_messages)
            results.append(temp.model_dump())


        # appending the follow up question but only for the simple plan thing
        # if temp_state == True:
        #     if isinstance(result, dict):
        #         result["follow_up"] = (
        #             "Do you need me to make a more detailed career plan for you?"
        #         )
        #     logger.info(" Added follow-up prompt for detailed plan offer.")

        store_career_plan(thread_id, results, is_review=False)

        current_results = state.get("task_results", {})
        new_results = {**current_results, "career_plan": results}

        return Command(
            goto="career_router",
            update={
                "task_results": new_results,
                "career_plan": results,
                "desired_job_role": temp_role,
                "simple": temp_state,
                "status": temp_status,
                # "messages": [AIMessage(content=json.dumps(result), name="CareerPlan")],
            },
        )
    except Exception as e:
        logger.error(
            f"Career plan creation failed for thread {thread_id}: {e}", exc_info=True
        )
        current_results = state.get("task_results", {})
        error_message = f"Error: Career plan could not be created. {e}"
        new_results_on_error = {
            **current_results,
            "career_plan": error_message,
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

# --- Replacement for general_qa_node ---

# --- REPLACE the entire general_qa_node function with this complete, more intelligent version ---

# --- REPLACE the entire general_qa_node function with this final, robust version ---


def general_qa_node(state: AgentState) -> Command:
    """
    Handles general Q&A using a forceful, keyword-driven framework for tool use.
    It prioritizes direct commands for web searches to ensure reliability.
    """
    thread_id = state["thread_id"]
    _log_transition(
        thread_id, "general_qa", "start", "Starting forceful, goal-aware general Q&A"
    )

    profile_data = state.get("profile_data")
    if not profile_data or "error" in profile_data:
        _log_transition(
            thread_id, "general_qa", "fail", "Profile data not available in state."
        )
        current_results = state.get("task_results", {})
        new_results_on_error = {
            **current_results,
            "general_qa": "Task failed: I need your profile data to answer questions contextually.",
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

    user_question = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_question = msg.content.strip()
            break

    if not user_question:
        current_results = state.get("task_results", {})
        new_results_on_error = {
            **current_results,
            "general_qa": "Task skipped: Could not identify a clear question.",
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )

    profile_context = json.dumps(profile_data, indent=2)
    plan_context = state.get("career_plan", "No career plan has been discussed yet.")
    job_description_context = state.get(
        "job_description", "No specific job goal has been set yet."
    )

    llm_with_tools = llm.bind_tools([web_search_tool])

    # Get recent context for understanding references
    # recent_ai_response = state.get("recent_ai_response", "")
    follow_up = state.get("follow_up", "")
    # logger.info(f"Recent AI response from general_qa_node: {recent_ai_response}")
    
    # --- THIS PROMPT USES AN UNBREAKABLE RULE TO FORCE TOOL USE ---
    qa_prompt = f"""You are an AI assistant with a web search tool as an addition. Your outputs are majorly for mobile phone users, so the outputs must be very short and crisp. 
    **A VERY IMPORTANT DISTINCTION TO ALWAYS UNDERSTAND**:
    - The user might not always talk about his profile or any career related question for that matter. The user might just want to have a conversation about any random thing in the world. IN THIS CASE, YOU MUST ACT AS A CAREER GUIDANCE BOT FROM **LearnTube.ai** (YOU MUST ALWAYS MAINTAIN THIS PERSONA) WHOSE CAPABILITY IS TO ENHANCE PROFILE, ANSWER GENERAL CAREER RELATED QUESTIONS, GIVE A CAREER PLAN IN ACCORDANCE OF THE USER'S PROFILE DATA. AND YOUR ANSWER MUST BE WIRED IN THIS TONE ONLY.  You must always consider the user's intent and context. You must always answer the user's question to the best of your ability. In that case always response in STRICTLY BETWEEN 50-70 WORDS ONLY. ALWAYS MAINTAIN A PROFESSIONAL TONE.
    
    **VERY IMPORTANT AND CRITICAL RULE TO FOLLOW (MUST FOLLOW)**:
    - ANY AND ALL OF THE OUTPUT YOU PROVIDE MUST BE STRICTLY WITHIN THE WORD LIMIT OF 160-180 WORDS ONLY. THIS MUST NOT BE EXCEEDED IN ANY SITUATION. THIS RULE MUST BE FOLLOWED ALL THE TIME. IT IS A STRICT AND CRITICAL BOUNDARY TO BE MAINTAINED. YOU MUST ALWAYS BE VERY PRCISE, CONCISE AND TO THE POINT, NO BEATING AROUND THE BUSH.
    
    You have 2 primary jobs:
    1.  Provide direct, precise and accurate answers to the user's questions, by taking into account all the context mentioned below, nothing less and nothing more.
    2.  THIS STEP IS NOT COMPULSARY - YOU SHOULD ONLY DO THIS IF YOU THINK PROVIDING WEB RESULTS WILL ANSWER THE USER'S QUESTION BETTER. IF THE USER ASKS FOR QUESTIONS RELATED TO THEIR PROFILE OR ANY OTHER GENERIC QUESTION THEN THIS ISN'T NEEDED. IF THE USER EXPLICITELY ASKS FOR WEB RESULTS THEN RUN THIS. Utilize the provided context to generate a list of web links that may or may not include the user's question. The generated web links or the search query used to generate the links should be directly related to the user's question and must answer what the user is trying to ask.

**CONTEXT:**
1.  **USER'S QUESTION:** "{user_question}"
2. USER'S DESIRED JOB ROLE: {state.get('desired_job_role')}
3.  **CURRENT CAREER PLAN UNDER DISCUSSION (Previous Turn):**
    {plan_context}
4.  **USER'S GOAL:** 
    {job_description_context}
5.  **USER'S PROFILE:**
    {profile_context}


6.  **LAST FOLLOW-UP QUESTION ASKED:** 
    {follow_up}

UNDERSTANDING USER'S REFERENCE:
- Always understand what the user is asking for by understanding the latest user message.
- If the user is making a reference to the previous response given, (understand that from the latest user message), then you MUST COMPULSARILY take the PREVIOUS AI RESPONSES into consideration and must use that to answer what the user is precisely asking for.
- Consider the LAST FOLLOW-UP QUESTION to understand if the user is responding to that specific question.

**CRITICAL DIRECTIVE: UNBREAKABLE TOOL-USE RULE**
Your reasoning process MUST follow this order:
1. VERY STRICT RULE: If a web search is required, produce exactly ONE tool call (web_search_tool) with a single search query string. Do not produce multiple tool calls. If you think the user's request covers multiple topics, synthesize them into one concise query that prioritizes the user's latest intent. This tool call must be done by keeping in mind all the below mentioned guidelines and rules by strictly following them..


MUST FOLLOW:
-MIND YOU TO NOT DO A WEB SEARCH FOR QUESTIONS THAT CAN BE ANSWERED WITH THE EXISTING CONTEXT. 
-THE DATA PROVIDED IN THE CONTEXT MUST ALWAYS BE USED TO ANSWER THE USER'S QUESTION. REFER TO THOSE WHEN QUESTIONS LIKE USER'S PROFILE QUESTIONS, GENERAL CAREER-RELATED QUESTIONS, QUESTIONS RELATED TO THE USER'S JOB ROLE OR PROFILE OR DESIRED JOB ROLE, QUESTIONS RELATED TO THE USER'S CAREER PLAN, QUESTIONS RELATED TO THE USER'S JOB DESCRIPTION, QUESTIONS RELATED TO THE USER'S AIMS AND FLAWS ETC ARE ASKED.

1. **Answering what the user is asking**: The user's question must be answered directly and accurately, using the context provided. The answer provided must be in context to what the user's profile is and what they are asking. Always maintain context of the user's profile and the context provided. The answer must be direct, precise and accurate. Do not add any extra information or advice. Do not make up any information. The questions can vary from simple to complex. They can be related to the user's profile, a short career related question related to the user's job role and/or profile, an email to a recruiter, a cover letter, a resume, interview preparation, skill development, job search strategies, career advancement, industry trends, or any other career-related topic. The answer must be in context to what the user's profile is and what they are asking ONLY, no additional information to be provided.

2. **Generating a list of web links**: 
    -ALWAYS THINK FROM A USER'S PERSPECTIVE. Put yourself in the user's shoes and think about what they are trying to ask. Then make a decision if a web search is necessary or not.
    -First, be mindful to perform a web search only if the user's question insinuates the need for additional and specific information that may not be present in the provided context. A WEB SEARCH MUST NOT BE PERFORMED FOR GENERIC QUESTIONS THAT CAN BE ANSWERED WITH THE EXISTING CONTEXT. UNECESSARY WEB SEARCHES SHOULD NOT BE DONE. 
    -IF THE QUESTION ASKED IS GENERIC AND CAN BE ANSWERED WITH THE EXISTING CONTEXT, THEN **DO NOT** PERFORM A WEB SEARCH. IF THE QUESTION IS SPECIFIC AND GIVING A WEB CONTEXT WILL HELP THE USER, THEN PERFORM A WEB SEARCH. HENCE ITS AN IMPORTANT DECISION TO MAKE.
    -Utilize the provided context to generate a list of web links that must answer what the user is asking. Remember, each search query or web link generated must be specifically related to the user's question and must answer what the user is trying to ask.  
    -If a web search is necessary, you MUST use the web_search_tool to perform the search. 


**CRITICAL FORMATTING RULE:**
-   Do **NOT** repeat the user's question in your response. Start directly with the answer. MIND YOU ALL THE RESPONSES GENERATED BY YOU, MUST BE IN THE STRICT WORD LIMIT OF 150-180 WORDS ONLY.
"""

    try:
        messages=[{"role": "system", "content": qa_prompt}]+state['messages']
        initial_response = llm_with_tools.invoke(messages)
        final_messages = [
            HumanMessage(content=qa_prompt, name="GeneralQAPrompt"),
            initial_response,
        ]

        if initial_response.tool_calls:
            _log_transition(
                thread_id,
                "general_qa",
                "tool_call",
                "Forcing web_search_tool based on unbreakable rule.",
            )
            tool_results_content = []
            for tool_call in initial_response.tool_calls:
                tool_result = web_search_tool.invoke(tool_call["args"])
                tool_results_content.append(json.dumps(tool_result))

            final_prompt = f"""A web search was performed and returned the following results:

SEARCH RESULTS:
{" ".join(tool_results_content)}

**Your Task:**
Format these results into a clean, user-friendly list of clickable Markdown links. Add a brief, one-sentence description for each link based on its title and snippet. Do not add any other conversational text. Just provide the list of links.
"""

            final_response = llm.invoke(final_prompt)
            result = final_response.content
        else:
            result = initial_response.content

        current_results = state.get("task_results", {})
        new_results = {**current_results, "general_qa": result}
        return Command(goto="career_router", update={"task_results": new_results})

    except Exception as e:
        _log_transition(thread_id, "general_qa", "fail", f"General Q&A failed for thread {thread_id}: {e}", exc_info=True)
        current_results = state.get("task_results", {})
        error_message = f"Error: I could not answer the question. {e}"
        new_results_on_error = {
            **current_results,
            "general_qa": error_message,
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )


# --- Replacement for user_interaction_node ---


def user_interaction_node(state: AgentState) -> Command:
    """Handles unclear requests, returns its result, and goes back to the router for housekeeping."""
    thread_id = state["thread_id"]
    _log_transition(
        thread_id, "user_interaction", "start", "Handling unclear interaction"
    )

    messages = state.get("messages", [])
    latest_user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_user_message = msg.content.strip()
            break

    clarification_prompt = f"""The user sent an unclear message: "{latest_user_message}"

Generate a brief, helpful clarification question. Suggest 2-3 specific actions they might want to take.
Keep response short and direct."""

    try:
        result = llm_call_with_retry_circuit(clarification_prompt).text
        current_results = state.get("task_results", {})
        new_results = {**current_results, "user_interaction": result}
        return Command(goto="career_router", update={"task_results": new_results})
    except Exception as e:
        logger.error(f"User interaction clarification failed: {e}")
        current_results = state.get("task_results", {})
        error_message = "I'm having trouble understanding."
        new_results_on_error = {
            **current_results,
            "user_interaction": error_message,
        }
        return Command(
            goto="career_router", update={"task_results": new_results_on_error}
        )


# --- Replacement for build_career_graph in agents.py ---


# --- FINAL, SIMPLE build_career_graph ---
def build_career_graph():
    """Builds the graph with the robust self-correction loop, using only simple edges."""
    global_logger.info("Building career assistance graph with Self-Correction Loop.")

    builder = StateGraph(AgentState)

    builder.add_node("analyze_profile", analyze_profile_node)
    builder.add_node("career_router", career_router_node)
    builder.add_node("synthesize_response", synthesize_response_node)
    builder.add_node("job_fit_analysis", job_fit_analysis_node)
    builder.add_node("enhance_profile", enhance_profile_node)
    builder.add_node("career_plan", career_plan_node)
    builder.add_node("general_qa", general_qa_node)
    builder.add_node("user_interaction", user_interaction_node)

    builder.set_entry_point("analyze_profile")
    builder.add_edge("analyze_profile", "career_router")

    # After ANY specialist worker node, control ALWAYS returns to the router for housekeeping.
    builder.add_edge("job_fit_analysis", "career_router")
    builder.add_edge("enhance_profile", "career_router")
    builder.add_edge("career_plan", "career_router")
    builder.add_edge("general_qa", "career_router")
    builder.add_edge("user_interaction", "career_router")

    # After synthesis, return to the router to wait for the next user input.
    builder.add_edge("synthesize_response", "career_router")

    global_logger.info("Career graph built successfully. Simple and robust.")
    return builder.compile(checkpointer=memory)


# DATA MANAGER
class ThreadDataManager:
    """Manages thread data in three MongoDB collections"""

    def __init__(self):
        self.profiles_collection = profiles_collection
        self.career_plan_collection = career_plan_collection
        self.conversations_collection = conversations_collection

        # Create indexes for efficient querying
        if profiles_collection is not None:
            try:
                profiles_collection.create_index("thread_id", unique=True)
                career_plan_collection.create_index("thread_id", unique=True)
                conversations_collection.create_index("thread_id")
                conversations_collection.create_index("timestamp")
                global_logger.info("MongoDB indexes created successfully")
            except Exception as e:
                global_logger.warning(f"Index creation failed: {e}")

    def get_thread_data(self, thread_id: str) -> dict:
        """Get complete thread data - legacy compatibility"""
        profile_data = self.get_profile_data(thread_id)
        return {"profile_data": profile_data} if profile_data else {}

    def get_profile_data(self, thread_id: str) -> dict:
        """Get profile data for thread"""
        return get_profile_data(thread_id)

    def get_conversation_history(self, thread_id: str) -> list:
        """Get conversation history for thread from conversations collection"""
        if self.conversations_collection is None:
            return []

        try:
            conversations = list(
                self.conversations_collection.find({"thread_id": thread_id}).sort(
                    "timestamp", 1
                )
            )

            return conversations
        except Exception as e:
            global_logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    def store_message(
        self, thread_id: str, node: str, role: str, content: str, metadata: dict = None
    ):
        """Store a conversation message"""
        return store_conversation_message(thread_id, node, role, content, metadata)


# Initialize components
career_graph = build_career_graph()
data_manager = ThreadDataManager()

# STREAMLIT INTEGRATION FUNCTIONS


def extract_ai_responses(messages: list) -> list:
    """Extract AI messages from message list for frontend display"""
    ai_responses = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content.strip():
            ai_responses.append(msg.content)
    return ai_responses


# Export main components
__all__ = [
    "career_graph",
    "memory",
    "data_manager",
    "extract_ai_responses",
    "linkedin_scraper_tool",
    "web_search_tool",
    "store_conversation_message",
    "get_thread_data",
    "store_thread_data",
    "store_profile_data",
    "get_profile_data",
    "store_career_plan",
    "get_career_plan",
    "find_existing_profile_by_linkedin_url",
    "get_thread_logger",
    "conversations_collection",
    "store_linkedin_url",
]

if __name__ == "__main__":
    global_logger.info("=== Enhanced Career Assistant with Tool Integration Ready ===")
    # global_logger.info(f"Graph nodes: {list(career_graph.get_graph().nodes.keys())}")
    # global_logger.info(f"MongoDB connected: {'Yes' if profiles_collection else 'No'}")
    global_logger.info("System ready with LangSmith tracing enabled")
    global_logger.info("Tools integrated: LinkedIn Scraper, Web Search")
    global_logger.info("Enhanced routing with detailed prompts implemented")
	
	