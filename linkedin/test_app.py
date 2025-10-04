# --- START OF FILE test_app.py ---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
import json
from agents import (
    career_graph, 
    conversations_collection, 
    store_linkedin_url, 
    find_existing_profile_by_linkedin_url,
    get_profile_data,
    get_thread_logger,
)
import uuid
import logging
from typing import List, Dict, Any


#for streaming and SSE
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Career Assistant API",
    description="API for career guidance and assistance",
    version="1.0.0"
)




# ---- Pydantic Models ----
class StartChatRequest(BaseModel):
    linkedin_url: str

class StartChatResponse(BaseModel):
    thread_id: str
    message: str
    status: str

class ResumeChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ResumeChatResponse(BaseModel):
    thread_id: str
    chat_history: List[ChatMessage]
    status: str

# ---- Utility ----
def fetch_chat_history(thread_id: str) -> List[Dict[str, str]]:
    try:
        if conversations_collection is None:
            return []
        docs = conversations_collection.find({"thread_id": thread_id}).sort("timestamp", 1)
        history = []
        for d in docs:
            # Map database roles to standard 'user' and 'assistant' roles for frontend
            role = "assistant" if d.get("role") in ["ai", "assistant"] else "user"
            history.append({
                "role": role, 
                "content": d.get("content", "")
            })
        return history
    except Exception as e:
        logger.error(f"Error fetching chat history for thread {thread_id}: {e}")
        return []

# ---- Endpoints ----

# --- MODIFIED: Reverted to synchronous 'def' ---
@app.post("/start-chat", response_model=StartChatResponse)
def start_chat(request: StartChatRequest):
    """Starts a new chat. The graph will run until it naturally interrupts at the career_router."""
    try:
        existing_profile = find_existing_profile_by_linkedin_url(request.linkedin_url)
        if existing_profile:
            thread_id = existing_profile.get('thread_id')
        else:
            thread_id = f"career-{uuid.uuid4().hex[:8]}"
            store_linkedin_url(thread_id, request.linkedin_url)

        # Setup thread-specific logging
        thread_logger = get_thread_logger(thread_id)
        
        config = {"configurable": {"thread_id": thread_id}}
        
        thread_logger.info(f" Invoking graph for session {thread_id}. It will run to the first natural interrupt.")
        
        # --- MODIFIED: Using synchronous 'invoke' ---
        career_graph.invoke({
            "messages": [{"role":"user", "content": "Hi"}],
            "thread_id": thread_id,
            "profile_data":{},
            "career_plan": "",
            "job_description": "",
            "desired_job_role": "",
            "profile_limitaions":"",
            "pending_tasks":[],
            "task_results":[],
            "is_planning":False,
            "task_in_progress":"",
            "simple":True,
            "analysis": False,
            "status": False,
            },
            config=config)

        # --- MODIFIED: Using synchronous 'get_state' ---
        current_state = career_graph.get_state(config)
        profile_data_in_state = current_state.values.get("profile_data")

        message = "Your profile has been analyzed and I'm ready for your command. What would you like to do?"
        
        if not profile_data_in_state or "error" in profile_data_in_state:
             message = "There may have been an issue analyzing your profile, but I am ready. What is your command?"

        return StartChatResponse(thread_id=thread_id, message=message, status="waiting_for_input")
    except Exception as e:
        # Use thread-specific logger if available, otherwise fall back to default logger
        if 'thread_id' in locals():
            thread_logger = get_thread_logger(thread_id)
            thread_logger.error(f" Error starting chat: {e}", exc_info=True)
        else:
            logger.error(f" Error starting chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start chat: {str(e)}")

#event generator for streaming in the resume point itself cuz resume payload n stuff is needed 
import asyncio

async def event_generator(thread_id: str, message: str):
    """Generator for SSE streaming"""
    thread_logger = get_thread_logger(thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    resume_payload = {"text": message}
    
    try:
        streamed_already = False
        
        # Stream from LangGraph
        for chunk in career_graph.stream(Command(resume=resume_payload), config=config, stream_mode="updates"):
            # Loop through all nodes in chunk
            for node_name, node_output in chunk.items():
                if not node_output or not isinstance(node_output, dict):
                    continue
                    
                # Check if this node updated messages
                if "messages" in node_output:
                    messages = node_output["messages"]
                    if not messages:
                        continue
                    
                    # Get last message
                    last_msg = messages[-1] if isinstance(messages, list) else messages
                    
                    # Stream only AI messages and only once (final synthesis)
                    if hasattr(last_msg, "type") and last_msg.type == "ai" and not streamed_already:
                        content = last_msg.content
                        thread_logger.info(f"Streaming from node '{node_name}'")
                        
                        # ← SEND START SIGNAL
                        yield f"data: {json.dumps({'type': 'start'})}\n\n"
                        
                        # Stream character by character
                        current = ""
                        for char in content:
                            current += char
                            yield f"data: {json.dumps({'content': current})}\n\n"
                            await asyncio.sleep(0.02)
                        
                        streamed_already = True
        
        # Done
        yield f"data: {json.dumps({'done': True})}\n\n"
        thread_logger.info(f"Streaming completed")
    except Exception as e:
        thread_logger.error(f" Streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"



# --- MODIFIED: Reverted to synchronous 'def' ---
@app.get("/resume-chat")
async def resume_chat(thread_id: str, message: str):
    """Resume chat with a new command using the Command(resume=...) pattern."""
    try:
        return StreamingResponse(
        event_generator(thread_id, message),
        media_type="text/event-stream"
    )
    except Exception as e:
        # Use thread-specific logger if available, otherwise fall back to default logger
        if 'thread_id' in locals():
            thread_logger = get_thread_logger(thread_id)
            thread_logger.error(f" Error resuming chat {thread_id}: {e}", exc_info=True)
        else:
            logger.error(f" Error resuming chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume chat: {str(e)}")


@app.get("/chat-history/{thread_id}", response_model=List[ChatMessage])
def get_chat_history(thread_id: str):
    try:
        # Setup thread-specific logging
        thread_logger = get_thread_logger(thread_id)
        thread_logger.info(f" Fetching chat history for thread_id: {thread_id}")
        history = fetch_chat_history(thread_id)
        return [ChatMessage(role=m["role"], content=m["content"]) for m in history]
    except Exception as e:
        # Use thread-specific logger if available, otherwise fall back to default logger
        if 'thread_id' in locals():
            thread_logger = get_thread_logger(thread_id)
            thread_logger.error(f"Error fetching chat history: {e}")
        else:
            logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {str(e)}")


@app.delete("/chat/{thread_id}")
def delete_chat(thread_id: str):
    try:
        # Setup thread-specific logging
        thread_logger = get_thread_logger(thread_id)
        thread_logger.info(f" Deleting chat for thread_id: {thread_id}")
        if conversations_collection is None:
            raise HTTPException(status_code=500, detail="Database not available")
        result = conversations_collection.delete_many({"thread_id": thread_id})
        thread_logger.info(f" Deleted {result.deleted_count} messages for thread_id: {thread_id}")
        return {"message": f"Chat {thread_id} deleted", "deleted_count": result.deleted_count, "status": "success"}
    except Exception as e:
        # Use thread-specific logger if available, otherwise fall back to default logger
        if 'thread_id' in locals():
            thread_logger = get_thread_logger(thread_id)
            thread_logger.error(f" Error deleting chat {thread_id}: {e}")
        else:
            logger.error(f" Error deleting chat: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")

# --- MODIFIED: Reverted to synchronous 'def' ---
@app.get("/profile/{thread_id}")
def get_profile_info(thread_id: str):
    """
    Fetches profile info directly from the graph's current state for consistency.
    """
    try:
        # Setup thread-specific logging
        thread_logger = get_thread_logger(thread_id)
        thread_logger.info(f" Fetching profile info from graph state for thread_id: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        
        # --- MODIFIED: Using synchronous 'get_state' ---
        current_state = career_graph.get_state(config)
        
        profile_data = current_state.values.get("profile_data")
        
        if not profile_data:
            thread_logger.warning(f"No profile data found in the graph state for thread_id: {thread_id}")
            return {"thread_id": thread_id, "profile_data": {}, "status": "no_profile_found_in_state"}
            
        thread_logger.info(f" Profile data found in state for thread_id: {thread_id}")
        return {"thread_id": thread_id, "profile_data": profile_data, "status": "success"}
    except Exception as e:
        # Use thread-specific logger if available, otherwise fall back to default logger
        if 'thread_id' in locals():
            thread_logger = get_thread_logger(thread_id)
            thread_logger.error(f" Error fetching profile info from state: {e}", exc_info=True)
        else:
            logger.error(f" Error fetching profile info from state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch profile info: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "AI Career Assistant API running"}		