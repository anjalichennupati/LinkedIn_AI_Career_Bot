
# %pip install -q -U google-generativeai beautifulsoup4 requests==2.32.3 PyPDF2
# %pip install -U langgraph langchain google-generativeai

# Imports
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # ✅ Fix here
from langchain_core.runnables import Runnable
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnableLambda


# ✅ 1. Imports and Gemini Setup
import requests
from bs4 import BeautifulSoup
import io
from PyPDF2 import PdfReader
# from google.colab import files
# from IPython.display import display

genai.configure(api_key="AIzaSyDX9htz1H9osUdf-RTN-z4DfiMLnbSUPkQ")
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")



from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

# 👇 Keep your reducer
def replace(_, b): return b

# ✅ Update the LangGraph state definition
class ProfileState(TypedDict):
    profile: Annotated[dict, replace]
    analysis: Annotated[str, replace]
    job_title: Annotated[str, replace]
    job_matches: Annotated[list, replace]
    job_fit_result: Annotated[str, replace]
    enhanced_content: Annotated[str, replace]
    career_guide_response: Annotated[str, replace]  # ✅ NEW



# # ✅ 2. Upload PDF and Parse Sections
# def upload_pdf():
#     print("📤 Upload your LinkedIn PDF")
#     uploaded = files.upload()
#     for fname in uploaded.keys():
#         print(f"\n✅ Uploaded: {fname}")
#         return io.BytesIO(uploaded[fname])
#     return None

def extract_profile_sections_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    sections = {
        "about": "",
        "experience": "",
        "skills": "",
        "education": "",
        "projects": "",
        "certifications": "",
        "publications": "",
        "volunteering": "",
        "awards": "",
        "languages": "",
        "courses": ""
    }

    current_section = None

    for line in full_text.splitlines():
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Detect section headers
        for section in sections.keys():
            if section in line_lower:
                current_section = section
                break
        else:
            if current_section:
                sections[current_section] += line_clean + " "

    # Final cleanup
    for key in sections:
        sections[key] = sections[key].strip()

    return sections



from langchain_core.runnables import RunnableLambda

def profile_analysis_agent(state: ProfileState) -> ProfileState:
    profile = state["profile"]

    prompt = """You are a professional AI Career Coach. You are given a LinkedIn profile parsed from a PDF.

Your task is to analyze each section for:
- ✅ Quality (strong/weak/missing)
- 🔍 Relevance to typical job roles
- 🛠 Suggestions to improve or rewrite the section

Return your output section-wise in a clear, structured format.

--- PROFILE START ---
"""
    for section in profile.keys():
        content = profile[section].strip()
        header = section.capitalize()
        if content:
            prompt += f"\n### {header}:\n{content}\n"
        else:
            prompt += f"\n### {header}:\n(Not Provided)\n"

    prompt += "\n--- PROFILE END ---\n\n"
    prompt += """
💡 For each section:
- Give a quality score: ✅ Strong / ⚠ Needs improvement / ❌ Missing
- Provide 2–3 suggestions to improve the section (content, phrasing, structure)
- Use clean formatting: *bold headings*, bullet points, and avoid unnecessary repetition.
"""

    response = model.generate_content(prompt)
    return {"profile": profile, "analysis": response.text}

# Wrap it as a Runnable
profile_analysis_node = RunnableLambda(profile_analysis_agent)



def search_remoteok_jobs(job_title, limit=15):
    search_term = job_title.lower().replace(" ", "-")
    url = f'https://remoteok.com/remote-{search_term}-jobs'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept-Language': 'en-US,en;q=0.9'
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    full_matches = []
    partial_matches = []

    for tr in soup.find_all('tr', class_='job'):
        try:
            title = tr.find('h2', itemprop='title').text.strip()
            company = tr.find('h3', itemprop='name').text.strip()
            location = tr.find('div', class_='location').text.strip() if tr.find('div', class_='location') else 'Remote'
            date_posted = tr.find('time')['datetime']
            link = 'https://remoteok.com' + tr['data-href']
            description = f"{title} at {company} in {location} — Posted on {date_posted}. More at: {link}"

            # Matching Logic
            title_lower = title.lower()
            job_title_lower = job_title.lower()

            if job_title_lower in title_lower or job_title_lower in description.lower():
                full_matches.append(description)
            elif any(word in title_lower for word in job_title_lower.split()):
                partial_matches.append(description)

        except Exception:
            continue

        if len(full_matches) >= limit:
            break

    return full_matches, partial_matches




from langchain_core.runnables import RunnableLambda

def job_fit_agent(state: ProfileState) -> ProfileState:
    profile = state["profile"]
    job_title = state["job_title"]

    # Get jobs from RemoteOK
    full_matches, partial_matches = search_remoteok_jobs(job_title)
    if not full_matches:
        return {
            "job_title": job_title,
            "job_matches": [],
            "job_fit_result": f"❌ No exact matches found for '{job_title}'. Try rephrasing."
        }

    job_text = "\n\n".join(full_matches[:8])[:3000]

    candidate_profile = f"""
About: {profile.get("about", "Not provided")}
Experience: {profile.get("experience", "Not provided")}
Skills: {profile.get("skills", "Not provided")}
"""
    for section in ["education", "projects", "certifications", "publications", "volunteering"]:
        content = profile.get(section, "").strip()
        if content:
            candidate_profile += f"{section.capitalize()}: {content}\n"

    prompt = f"""
You are a Job Fit Analysis AI.

--- CANDIDATE PROFILE ---
{candidate_profile}

--- JOB ROLE TARGETED ---
"{job_title}"

--- LIVE JOB LISTINGS CONTAINING FULL JOB TITLE (total: {len(full_matches)}) ---
{job_text}

💡 Task:
1. Evaluate the candidate’s fitness for this job role based on the provided job listings and their full profile.
2. Provide a **Job Match Score out of 100**, clearly explained.
3. List 4-6 strengths from the candidate that align well with the jobs.
4. Suggest 4-6 specific improvements in skills, experience, or phrasing that would increase the match score.
5. Only consider jobs that fully match the title.
6. Present your answer in a clean, readable format with bullet points and bold headings where needed.
"""

    response = model.generate_content(prompt)

    return {
        "job_title": job_title,
        "job_matches": full_matches,
        "job_fit_result": response.text
    }

# Wrap it
job_fit_node = RunnableLambda(job_fit_agent)



def content_enhancement_agent(state: ProfileState) -> ProfileState:
    profile = state["profile"]
    job_title = state["job_title"]
    analysis = state["analysis"]
    job_matches = state.get("job_matches", [])

    job_context = "\n\n".join(job_matches[:5])[:3000] if job_matches else ""

    prompt = f"""
You are a LinkedIn Profile Rewriter AI helping a candidate optimize their profile for the job title: "{job_title}".

You are given:
1. Their **current profile** (parsed from PDF).
2. A **profile review analysis** of their content.
3. 3–5 **real job listings** for the target role.

--- JOB LISTINGS ---
{job_context or 'Not Available'}

--- PROFILE ANALYSIS ---
{analysis}

--- CURRENT PROFILE ---
"""
    for section in profile:
        content = profile[section].strip()
        prompt += f"\n## {section.capitalize()}:\n{content if content else '(Not provided)'}"

    prompt += """

💡 Now rewrite/improve the **full profile** section-wise:
- For each section (About, Experience, Skills,Education, Publications, Certifications etc.), generate a **refined version** tailored to the job.
- Improve language, formatting, and alignment to industry keywords.
- Do NOT invent experience; only rephrase what’s there.
- Use clean markdown formatting: bold headings, bullet points, short paragraphs.

📌 Output format should be:

### About:
<rewritten about>

### Experience:
<rewritten experience>

... and so on for each section.
"""

    response = model.generate_content(prompt)

    return {
        "profile": profile,
        "job_title": job_title,
        "job_matches": job_matches,
        "analysis": analysis,
        "enhanced_content": response.text
    }


content_enhancement_node = RunnableLambda(content_enhancement_agent)




def career_guide_agent(state: ProfileState) -> ProfileState:
    import re

    profile = state.get("profile", {})
    analysis = state.get("analysis", "")
    job_title = state.get("job_title", "")
    job_matches = state.get("job_matches", [])
    job_fit_result = state.get("job_fit_result", "")
    enhanced_content = state.get("enhanced_content", "")
    user_query = state.get("user_question", "What should I improve in my profile?")

    # ✅ Normalize question for routing
    query = user_query.lower()

    # ✅ Include relevant sections conditionally
    include_profile = True
    include_analysis = any(k in query for k in ["career", "improve", "switch", "skills", "certification", "path"])
    include_enhanced = any(k in query for k in ["rewrite", "update", "enhance", "linkedin", "resume", "cv"])
    include_jobs = any(k in query for k in ["recruiter", "email", "fit", "jd", "job", "apply"])
    include_jobfit = any(k in query for k in ["match", "qualified", "skills", "experience", "alignment"])

    # ✅ Utility: safe trim
    def trim(text, max_chars=1500):
        return text[:max_chars] if text else "N/A"

    # ✅ Clean JD formatting
    job_text = "\n\n".join([f"• {re.sub(r'\s+', ' ', job)[:400]}" for job in job_matches[:3]]) if include_jobs else ""

    # ✅ Assemble final context only with relevant parts
    context_parts = [
        f"--- USER QUESTION ---\n{user_query}",
        f"--- ORIGINAL PROFILE ---\n{trim(profile.get('about', 'N/A'), 1000)}" if include_profile else "",
        f"--- ANALYSIS ---\n{trim(analysis)}" if include_analysis else "",
        f"--- JOB TITLE TARGETED ---\n{job_title}" if job_title else "",
        f"--- ENHANCED CONTENT ---\n{trim(enhanced_content)}" if include_enhanced else "",
        f"--- TOP JOB LISTINGS ---\n{job_text}" if job_text else "",
        f"--- JOB FIT ANALYSIS ---\n{trim(job_fit_result)}" if include_jobfit else "",
    ]

    full_context = "\n\n".join([part for part in context_parts if part.strip()])

    # ✅ Smart prompt
    prompt = f"""
You are a smart and experienced AI Career Advisor.

Your job is to:
1. Understand the user's question.
2. Use only the **relevant context** from the profile, job analysis, and job descriptions.
3. Respond with clear, practical, personalized advice.

Instruction examples:
- If the question is about writing to a recruiter → create a compelling cold email.
- If the question is about a career path → outline next 2–3 roles with required skills.
- If it's about switching fields → suggest adjacent roles with transferable skills.
- If it's about profile improvement → give 3–4 specific improvements with reasoning.
- If the question is unclear or seems mismatched with the current profile:
    - Try to infer the user's intent from the available context.
    - Provide the best personalized advice possible, even if it’s general.
    - Only ask clarifying questions if no useful response can be given.


{full_context}
"""

    try:
        response = model.generate_content(prompt)
        return {**state, "career_guide_response": response.text}
    except Exception as e:
        return {**state, "career_guide_response": f"❌ Error generating response: {e}"}


career_guide_node = RunnableLambda(career_guide_agent)





# ✅ Create the graph
memory = MemorySaver()
builder = StateGraph(ProfileState)

# Add all three nodes
builder.add_node("profile_analysis", profile_analysis_node)
builder.add_node("job_fit", job_fit_node)
builder.add_node("content_enhancement", content_enhancement_node)
builder.add_node("career_guide", RunnableLambda(career_guide_agent))


# Define flow
builder.set_entry_point("profile_analysis")
builder.add_edge("profile_analysis", "job_fit")
builder.add_edge("job_fit", "content_enhancement")
builder.add_edge("content_enhancement", "career_guide")
builder.set_finish_point("career_guide")


# Compile
graph = builder.compile()




