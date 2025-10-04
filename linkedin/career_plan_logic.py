from langchain_core.messages import AIMessage, HumanMessage
from agents import (
    llm_call_with_retry_circuit,
    websearch_mcp_node,
    enrich_websearch_node,
    store_websearch_node,
    AgentState,
    interrupt,
)


def career_plan_node(state: AgentState) -> dict:
    profile = state.get("profile_data", {})
    messages = state["messages"]

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
Return JSON: {"use_web": true/false, "confidence": 0-100, "reason": "..."}  
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
