# linkedin_scraper.py
from apify_client import ApifyClient
import os
from dotenv import load_dotenv

load_dotenv()
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

client = ApifyClient(APIFY_TOKEN)

# def scrape_linkedin_profile(profile_url):
#     run_input = {
#         "profileUrls": [profile_url],
#         "maxProfiles": 1
#     }

#     try:
#         run = client.actor("apify/actor-scraper").call(run_input=run_input)
#         output = run.get("output", [])
        
#         # Check for Apify-specific error responses
#         if output and "error" in output[0]:
#             return {"error": output[0]["error"]}
        
#         return output[0] if output else {"error": "No data returned from Apify."}
    
#     except Exception as e:
#         return {"error": str(e)}


def scrape_linkedin_profile(profile_url):
    # MOCKED PROFILE DATA for testing
    return {
        "about": "Aspiring AI/ML engineer with experience in NLP, deep learning, and big data pipelines. Worked on LLM-based ad optimization and generative apps.",
        "experience": "AI Intern at ClearSpot.ai, DRDO intern in signal processing, Led GenAI and Blockchain projects.",
        "skills": "Python, LangChain, OpenAI API, FastAPI, AWS, Streamlit, Scikit-learn, React, Node.js, MongoDB"
    }
