# 🔗 LinkedIn Career Coach AI (Multi-Agent System)

> 🎯 An intelligent, interactive career guidance system powered by **LangGraph**, **Gemini AI**, and **Streamlit** that analyzes LinkedIn profiles, evaluates job-fit, rewrites content, and offers personalized career advice.

---

## 🚀 Overview

**LinkedIn Career Coach AI** is a chat-based multi-agent platform designed to guide users in:

* Optimizing their LinkedIn profiles,
* Evaluating their readiness for specific job roles,
* Enhancing content alignment with job descriptions,
* And offering strategic, personalized career mentorship.

Built using **Google Gemini**, **LangGraph**, and **Streamlit**, this system simulates an AI career coach that can deeply understand your professional background and offer dynamic, contextual recommendations based on real job listings.

Demo Video link: https://drive.google.com/file/d/1blflFWqw3E-9eAdoSDwz3DyXWb72CoUR/view?usp=sharing
---

## 🧠 Features & Agent Functions

### 1️⃣ Profile Analysis Agent

* Parses uploaded **LinkedIn PDF** profiles.
* Evaluates each section: About, Experience, Skills, Projects, etc.
* Returns:

  * Section-wise quality score (✅ Strong, ⚠ Needs Improvement, ❌ Missing)
  * Suggestions for structure and clarity.
  * Highlights gaps in standard roles.

---

### 2️⃣ Job Fit Analysis Agent

* Scrapes **live job listings** from [RemoteOK](https://remoteok.com) using the target job title.
* Compares profile content with 3–5 real job descriptions.
* Generates:

  * Job Match Score out of 100
  * 3 profile strengths & 3 areas for improvement
  * Job listing suggestions

---

### 3️⃣ Content Enhancement Agent

* Takes the job listings + profile analysis.
* Rewrites key sections (**About, Experience, Skills**) to:

  * Match industry-standard phrasing.
  * Include relevant keywords from top job listings.
  * Increase ATS visibility and recruiter appeal.

---

### 4️⃣ Career Guide Agent

* Smart AI chatbot using **contextual routing** (router agent).
* Accepts any career-related question and responds accordingly:

  * Cold email writing ✉️
  * Career growth path 🧭
  * Skill gap identification 🎓
  * Job switch advice 🔁
* Uses previous outputs: enhanced profile, job fit analysis, and user query to tailor its response.

---

## 🧱 System Architecture


<p align="center">
  <img src="[https://i.postimg.cc/pLDYXr3P/dee.png](https://i.postimg.cc/7Y8bj6jb/Whats-App-Image-2025-06-30-at-15-32-08-871731c6.jpg)" alt="App Screenshot" width="400">
</p> 

> All agents share state via LangGraph’s state manager and memory system (`MemorySaver`) for full context awareness across the pipeline.


<p align="center">
  <img src="https://i.postimg.cc/pLDYXr3P/dee.png" alt="App Screenshot" width="600">
</p>  


---

## 🛠 Tech Stack

| Layer        | Tool/Library                  | Purpose                                     |
| ------------ | ----------------------------- | ------------------------------------------- |
| 🧠 AI Models | **Gemini 1.5 Flash**          | Language understanding and generation       |
| 🕸 Backend   | **LangGraph**, **LangChain**  | Multi-agent system, routing, state handling |
| 🗃 Parsing   | **PyPDF2**                    | Extract structured text from PDF profiles   |
| 🌐 Scraping  | **BeautifulSoup**, `requests` | Fetch job listings from RemoteOK            |
| 🌈 Frontend  | **Streamlit**                 | Interactive web app with tab layout         |
| 💾 Memory    | **LangGraph MemorySaver**     | Session + state memory across agents        |

---

## 📂 Repository Structure

```
📁 linkedin-career-coach/
│
├── agents.py                # All LangGraph agent logic (4 agents)
├── app.py                   # Streamlit frontend with tabbed UI
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── sample_profile.pdf       # Test LinkedIn profile PDF
```

---

## ⚙️ Installation & Setup

### ✅ Requirements

* Python 3.8+
* Streamlit
* Google Gemini API Key (free tier is enough to start)

---

### 🧾 1. Clone the Repo

```bash
git clone https://github.com/your-username/linkedin-career-coach.git
cd linkedin-career-coach
```

---

### 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

OR individually:

```bash
pip install streamlit langgraph langchain google-generativeai beautifulsoup4 requests PyPDF2
```

---

### 🔑 3. Setup Gemini API Key

In `agents.py`, configure:

```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY_HERE")
```

You can get one from [Gemini Console](https://ai.google.dev).

---

### ▶️ 4. Run the App

```bash
streamlit run app.py
```

Then open in your browser at `http://localhost:8501`

---

## 🎯 Usage Walkthrough

1. **Upload your LinkedIn PDF**
2. **Enter target job title** (e.g., "Data Scientist")
3. Use the tabs:

   * `Profile Analysis`: Analyze profile strength
   * `Job Fit`: Match with real job listings
   * `Content Enhancement`: Rewrite profile sections
   * `Career Guide`: Ask anything — from email help to roadmap advice
   
   

---


