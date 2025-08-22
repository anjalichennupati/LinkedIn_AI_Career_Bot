# career_plan_mcp.py
from fastmcp import FastMCP

mcp = FastMCP("Career Plan MCP")


@mcp.tool()
def career_plan(profile: dict, prompt: str) -> dict:
    """
    Generates a career plan given profile JSON + short user prompt.
    """
    return {
        "plan": [
            f"Step 1: Use {profile.get('skills', [])} to build projects",
            f"Step 2: Focus on {prompt}",
            "Step 3: Network + apply for internships",
        ]
    }


if __name__ == "__main__":
    mcp.run()
