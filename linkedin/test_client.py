import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys, os

server_file = os.path.abspath("E:\\LinkedIn_AI_Career_Bot - Copy\\linkedin\\career_plan_mcp.py")

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_file],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            response = await session.list_tools()
            print("\nTools:")
            for tool in response.tools:
                print(tool.name)

            # Call your tool
            result = await session.call_tool(
                "generate_career_plan",
                arguments={
                    "profile_data": {"name": "Bro", "field": "AI/ML"},
                    "messages": [{"role": "user", "content": "I want a career plan for AI"}],
                },
            )
            print("\nCareer Plan Output:")
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())
