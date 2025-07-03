# AI Career Assistant – Chat-based LinkedIn Profile Analyzer

## Overview

This project is an interactive, AI-powered Career Assistant that analyzes LinkedIn profiles (in PDF format), provides personalized feedback, evaluates job fit for custom job descriptions, and suggests profile enhancements—all through a conversational chatbot interface. 

It leverages **Google Gemini's Generative AI** to process user inputs and orchestrates its logic flow using **LangGraph**, a declarative workflow framework built on top of LangChain. The front-end interface is built using **Streamlit**, enabling a lightweight web app experience for end users.

Users can upload a LinkedIn PDF profile, optionally paste a job description, and ask career-related questions.

The system processes the data intelligently and generates detailed insights using LLM-powered agents.

Demo Video: https://drive.google.com/file/d/1VbZ2lk20VnZx3R2gcCD64u6rVpI3m-fx/view?usp=sharing
Website: https://linkedinaicareerbot-vgx9ib4m9zgdemctuufbke.streamlit.app/

---

## Features

- **PDF Profile Parsing**: Automatically extracts key sections like About, Skills, Experience, Education, etc. from a LinkedIn profile PDF.
- **Career Analysis Agent**: Evaluates strengths, weaknesses, and suggests improvement tips based on profile content.
- **Job Fit Evaluator**: Scores a user's profile against a provided job description and suggests ways to better align the two.
- **Profile Enhancement Agent**: Rewrites and optimizes sections of the user's profile to increase alignment with industry standards or a specific job role.
- **Conversational Interface**: Users can ask career-related questions, and the system responds in a helpful and personalized tone.
- **Persistent Conversation History**: Keeps track of previous questions and answers within a session.

---

## System Architecture

The system is composed of the following layers:

### 1. Input Layer
- Users upload a **PDF** of their LinkedIn profile.
- Optionally, users can paste a **job description**.
- They can also enter any **free-form career question**.

### 2. Profile Extraction Layer
- The `PyPDF2` library is used to extract raw text from the uploaded PDF.
- Section headers such as "Experience", "Education", etc. are identified and organized into a structured dictionary format.

### 3. LangGraph Agents
- A `ProfileState` data structure (using `TypedDict`) tracks inputs and outputs across stages.
- The LangGraph pipeline includes:
  - `analyze_profile`: Extracts strengths, weaknesses, and suggestions.
  - `job_fit_agent`: Matches profile to job description and provides a score.
  - `enhance_profile`: Improves and rewrites profile content.
  - `career_qa_agent`: Routes and merges responses for the final answer.

### 4. LLM Backend
- The app uses **Google Gemini 1.5 Flash** via the `google.generativeai` Python SDK.
- Prompts are dynamically generated based on user inputs and system state.

### 5. Frontend (Streamlit)
- Streamlit UI handles file uploads, text input, and displays chat history.
- A spinner shows the thinking state during response generation.
- Session state is used to preserve the chat thread and profile data.

---

## Tech Stack

| Category          | Library / Platform                  |
|-------------------|--------------------------------------|
| Programming Language | Python 3.9+                         |
| LLM API           | Google Generative AI (Gemini 1.5 Flash) |
| Orchestration     | LangGraph                            |
| Prompt Execution  | LangChain Core                       |
| PDF Parsing       | PyPDF2                               |
| Frontend Framework| Streamlit                            |
| Memory Handling   | LangGraph MemorySaver                |
| Session Management| Streamlit Session State              |
| Version Control   | Git + GitHub                         |

---

## Installation

To run this project locally, follow the steps below:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-career-assistant.git
cd ai-career-assistant
