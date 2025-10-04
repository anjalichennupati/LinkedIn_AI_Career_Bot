# career_plan_mcp.py
from fastmcp import FastMCP, tools
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import traceback
import json # Added for better formatting of web results in the prompt

# Import your helpers
from agents import (
    
    llm_call_with_retry_circuit,
)

mcp = FastMCP("career-plan-service")


@mcp.tool()
def generate_career_plan(
    profile_data: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Generate a career plan given profile data and chat history.
    This MCP tool encapsulates the original career_plan_node logic and
    now directly integrates web search based on the user's query.
    """

    # Convert message dicts -> objects
    parsed_messages = []
    for m in messages:
        if m.get("role") == "user":
            parsed_messages.append(HumanMessage(m["content"]))
        elif m.get("role") == "assistant":
            parsed_messages.append(AIMessage(m["content"]))

    # Find latest user message to use as the search query
    latest_user_msg = None
    for msg in reversed(parsed_messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content.strip()
            break

    if not latest_user_msg:
        return {
            "plan_output": "âš  Please provide a brief or question for your career plan.",
            "web_used": False,
        }

    # Format profile text
    profile_text = (
        "\n".join(f"{k}: {v}" for k, v in profile_data.items())
        if profile_data
        else "(no profile)"
    )

    # --- Integrated Web Search Step ---
    # Directly search the web using the latest user message as the query.
    enriched_results = []
    web_info_for_prompt = ""
    


    # --- Generate Career Plan Step ---
    # This prompt is kept the same as you requested.
    plan_prompt = f"""
    User Input: {latest_user_msg}

    Profile Data:
    {profile_text}

    {'Additional Web Info:\n' + web_info_for_prompt if web_info_for_prompt else ''}

    You are an expert AI career coach with 10+ years of experience helping professionals advance their careers.

INSTRUCTIONS:
1. Create a specific, actionable career plan based on the user's profile and request
2. Include concrete steps with timelines where appropriate
3. If web research is available, prominently feature the specific courses, certifications, and resources found
4. Address the user's specific situation and goals from the conversation history
5. Provide realistic milestones and success metrics
6. Include both technical and soft skill development where relevant
7. Maintain context from previous conversation - don't lose track of their original goals

Generate a comprehensive career plan that is tailored to this individual's profile and goals.
    """.strip()

    try:
        result = llm_call_with_retry_circuit(plan_prompt)
        plan_output = result.text
    except Exception as e:
        plan_output = f"âŒ Error generating career plan: {e} {traceback.format_exc()}"

    return {
    "plan_output": plan_output,
    "web_used": False
}



if __name__ == "__main__":
    mcp.run()