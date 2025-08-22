# career_plan_mcp.py
from fastmcp import FastMCP, tools
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Import your helpers
from agents import (
    websearch_mcp_node,
    enrich_websearch_node,
    store_websearch_node,
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
    This MCP tool encapsulates the original career_plan_node logic.
    """

    # Convert message dicts -> objects
    parsed_messages = []
    for m in messages:
        if m.get("role") == "user":
            parsed_messages.append(HumanMessage(m["content"]))
        elif m.get("role") == "assistant":
            parsed_messages.append(AIMessage(m["content"]))

    # Find latest user message
    latest_user_msg = None
    for msg in reversed(parsed_messages):
        if isinstance(msg, HumanMessage):
            latest_user_msg = msg.content.strip()
            break

    if not latest_user_msg:
        return {
            "plan_output": "⚠ Please provide a brief or question for your career plan.",
            "web_used": False,
            "search_summaries": [],
        }

    # Format profile text
    profile_text = (
        "\n".join(f"{k}: {v}" for k, v in profile_data.items())
        if profile_data
        else "(no profile)"
    )

    # Step 1: Ask LLM if web info is needed
    web_check_prompt = f"""
    User Input: {latest_user_msg}

    Profile Data:
    {profile_text}

    TASK:
    Decide whether fetching external web info (like courses, communities, websites, industry resources) is required.  
    Return JSON: {{ "use_web": true/false, "confidence": 0-100, "reason": "..." }}
    """.strip()

    try:
        web_decision = (
            llm_call_with_retry_circuit(web_check_prompt).text.strip().lower()
        )
    except Exception as e:
        web_decision = "no"

    # Step 2: If yes → web search nodes
    search_summaries = []
    if "true" in web_decision or "yes" in web_decision:
        try:
            state = {"profile_data": profile_data, "messages": parsed_messages}
            state = websearch_mcp_node(state)
            state = enrich_websearch_node(state)
            state = store_websearch_node(state)

            search_results = state.get("websearch_summary", [])
            if search_results:
                search_summaries = [
                    {"title": r["title"], "link": r["link"]} for r in search_results
                ]
        except Exception as e:
            search_summaries = [{"error": str(e)}]

    # Step 3: Generate career plan
    plan_prompt = f"""
    You are an AI career coach.

    User Input: {latest_user_msg}

    Profile Data:
    {profile_text}

    {"Additional Web Info:\n" + str(search_summaries) if search_summaries else ""}

    TASK:
    - Generate a clear, actionable career plan or revise an existing one based on user input.
    - Integrate relevant insights from web results if available.
    - Provide practical steps, resources, and guidance.
    Respond concisely and practically.
    """.strip()

    try:
        result = llm_call_with_retry_circuit(plan_prompt)
        plan_output = result.text
    except Exception as e:
        plan_output = f"❌ Error generating career plan: {e}"

    return {
        "plan_output": plan_output,
        "web_used": bool(search_summaries),
        "search_summaries": search_summaries,
    }


if __name__ == "__main__":
    mcp.run()
