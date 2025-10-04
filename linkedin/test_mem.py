# career_plan_mcp_with_procedural.py
from fastmcp import FastMCP, tools
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage

# LangGraph + LangMem
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langmem import create_prompt_optimizer

# ===== Define your MCP service =====
# mcp = FastMCP("career-plan-service")

# Career Plan Tool (replace email draft tool)
# @mcp.tool()
def career_plan_tool(user_profile: Dict[str, Any]) -> str:
    """
    Generate a personalized career plan.
    """
    # This is where your career plan logic goes
    return f"Career plan generated for {user_profile.get('name', 'User')} 🚀"

# ===== Procedural Memory Setup =====
store = InMemoryStore(get_store())
sys_prompt = {"role": "system", "content": "You are an AI career coach. Help users plan their career path step by step."}

agent = create_react_agent(
    prompt=sys_prompt,
    tools=[career_plan_tool],
    store=store,
)

# ===== Prompt Optimizer =====
optimizer = create_prompt_optimizer()

# Example feedback trajectory
feedback = {
    "request": "Always suggest at least 3 career paths, with pros/cons."
}

optimizer_result = optimizer.optimize(
    current_prompt=sys_prompt["content"],
    trajectories=feedback,
)

print("Optimized Prompt:", optimizer_result)

# ===== Example Run =====
if __name__ == "__main__":
    # Simulate a user asking for a career plan
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "I am a CS student interested in AI and blockchain. Suggest a career plan."}
        ]
    })
    print(result["messages"][-1]["content"])
