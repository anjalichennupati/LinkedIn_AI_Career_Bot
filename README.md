# 💼 AI Career Coaching Assistant with LangGraph + Gemini

An interactive, intelligent career QA agent built using LangGraph, Google Gemini 1.5 Flash and Streamlit.  
It provides career guidance, resume audits, profile enhancement, and job-fit analysis — all via natural conversations.

---

##  Overview

This project demonstrates a **modular, multi-agent system** using the LangGraph framework for fine-grained, dynamic routing of career-related queries. It maintains persistent memory, allows context-aware rerouting, and supports human-in-the-loop querying (interrupt and resume).

Users can:

- Get an audit of their LinkedIn profile.
- Match their skills to job descriptions.
- Improve specific profile sections.
- Ask general career development questions.

---

##  Features

###  Core Functionalities

| Feature                          | Description |
|----------------------------------|-------------|
|  **Profile Analysis**         | Provides detailed strengths, weaknesses, and section ratings for a LinkedIn/resume profile. |
|  **Job Match Evaluation**     | Compares your profile with a job description and gives a match score, improvement tips, and a verdict. |
|  **Resume/LinkedIn Enhancer**| Rewrites About, Experience, Projects, and other sections using AI best practices and alignment with target roles. |
|  **General Career Q&A**       | Answers anything loosely related to career paths, switching domains, skill-building, certifications, etc. |
|  **Live LinkedIn Scraping**   | Parses profile data from a public LinkedIn URL using the Apify API. |
|  **Interrupt + Resume Chat**  | Maintains chat flow and context using LangGraph’s built-in memory. |

---

###  LangGraph Features Used

| LangGraph Feature       | Purpose                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| `StateGraph`             | Graph construction and node routing                                     |
| `Command()`              | For LLM-based routing to specific nodes                                 |
| `interrupt()`            | To pause execution for user input (human-in-the-loop mode)              |
| `operator.add`           | Merges and accumulates the chat message list persistently               |
| Custom TypedDict State   | Stores messages, job description, and profile data                      |
| Circular Flow Design     | All task nodes return to the router node for continuous conversation    |

---

##  System Architecture (Detailed)

###  High-Level Workflow

1. The user types a message.
2. If it includes a **LinkedIn URL** or **job description**, it's captured and stored in memory.
3. Otherwise, the message is passed to the **Router Node**, which uses LLM-based natural language routing to determine intent:
    - Analyze profile?
    - Compare to job description?
    - Rewrite/improve content?
    - General question?
4. The appropriate agent is invoked.
5. After response, flow **returns to router** for the next question.
6. If the user types `quit`, the conversation ends.

###  LangGraph Node Map

| Node Name             | Role |
|------------------------|------|
| `career_qa_router`     | LLM-based intent router |
| `analyze_profile`      | LinkedIn/profile auditor |
| `job_fit_agent`        | JD-matching evaluator |
| `enhance_profile`      | AI-based profile rewriter |
| `general_qa_node`      | Handles all out-of-scope general questions |
| `END`                  | Terminates chat loop |

---

###  Block Diagram (LangGraph Execution Flow)

```mermaid
flowchart TD
    UserInput([" User Input"]) --> Router

    subgraph LangGraph Nodes
        Router[" career_qa_router"]
        Analyze[" analyze_profile"]
        JobFit[" job_fit_agent"]
        Enhance[" enhance_profile"]
        GeneralQA[" general_qa_node"]
        EndNode([" END"])
    end

    Router --> |"input includes LinkedIn"| Router
    Router --> |"input includes Job Description"| Router

    Router --> |"LLM decides: analyze_profile"| Analyze
    Router --> |"LLM decides: job_fit_agent"| JobFit
    Router --> |"LLM decides: enhance_profile"| Enhance
    Router --> |"LLM decides: general_qa"| GeneralQA
    Router --> |"input = quit/exit/stop"| EndNode

    Analyze --> Router
    JobFit --> Router
    Enhance --> Router
    GeneralQA --> Router
