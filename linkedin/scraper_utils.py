# scraper_utils.py

from apify_client import ApifyClient

import os
from dotenv import load_dotenv

load_dotenv()  


def scrape_and_clean_profile(linkedin_url: str, api_token: str) -> dict:
    client = ApifyClient(api_token)

    run_input = { "queries": [linkedin_url] }
    run = client.actor("harvestapi~linkedin-profile-scraper").call(run_input=run_input)

    raw = None
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        raw = item.get("element", {})
        break

    if not raw:
        return {}

    clean_data = {
        "name": f"{raw.get('firstName', '')} {raw.get('lastName', '')}".strip(),
        "headline": raw.get("headline", ""),
        "about": raw.get("about", ""),
        "location": raw.get("location", {}).get("linkedinText", ""),
        "connections": str(raw.get("connectionsCount", "")),
        "experience": "\n".join(
            f"{exp.get('position', '')} at {exp.get('companyName', '')} ({exp.get('duration', '')})\n{exp.get('description', '')}"
            for exp in raw.get("experience", [])
        ),
        "education": "\n".join(
            f"{edu.get('degree', '')} in {edu.get('fieldOfStudy', '')} at {edu.get('schoolName', '')} ({edu.get('period', '')})"
            for edu in raw.get("education", [])
        ),
        "skills": ", ".join(skill.get("name", "") for skill in raw.get("skills", [])),
        "certifications": "\n".join(
            f"{cert['title']} ({cert['issuedAt']})"
            for cert in raw.get("certifications", [])
        ),
        "publications": "\n".join(
            f"{pub['title']} - {pub.get('description', '')}"
            for pub in raw.get("publications", [])
        ),
    }
    return clean_data
