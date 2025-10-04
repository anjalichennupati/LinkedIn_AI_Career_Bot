# career_plan_mcp.py
from fastmcp import FastMCP, tools
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage



# Import MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient
# install (only once in env)
# pip install -U langmem langgraph

from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langmem import create_prompt_optimizer
from langchain_core.tools import tool


# Procedural memory store
store = InMemoryStore()
key = "agent_instructions"







client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Make sure to update to the full absolute path to your math_server.py file
            "args": ["linkedin/career_plan_mcp.py"],
            "transport": "stdio",
        },
        "weather": {
            # Make sure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    }
)

tools = await client.get_tools()



def career_prompt(state):
    item = store.get(("instructions",))
    instructions = item.value if item else "You are an AI career coach."
    sys_prompt = {"role": "system", "content": instructions}
    return [sys_prompt] + state["messages"]


agent = create_react_agent(
    model="models/gemini-1.5-flash",
    prompt=career_prompt,
    tools=tools,
    store=store,
)


optimizer = create_prompt_optimizer()

# Example feedback to refine career plan outputs
feedback = {
    "request": "Always include at least 3 possible career paths with pros/cons."
}

current_prompt = store.get(("instructions",))
optimizer_result = optimizer.invoke(
    {"prompt": current_prompt, "trajectories": [(None, feedback)]}
)

# Update memory with optimized prompt
store.put(("instructions",), optimizer_result)


