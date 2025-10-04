from mcp.server.fastmcp import FastMCP
from scraper_utils import scrape_and_clean_profile
import os

mcp = FastMCP("LinkedInScraper")


@mcp.tool()
async def linkedin_scraper(linkedin_url: str) -> dict:
    """Scrape LinkedIn profile data."""
    if not linkedin_url:
        return {"error": "No LinkedIn URL provided"}

    api_token = os.getenv("APIFY_API_TOKEN")
    profile_data = scrape_and_clean_profile(linkedin_url, api_token)
    return {"profile_data": profile_data}


if __name__ == "__main__":
    print("🚀 LinkedIn Scraper FastMCP running on http://localhost:8000")
    mcp.run(transport="streamable-http")
