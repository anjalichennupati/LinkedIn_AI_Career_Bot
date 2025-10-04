# Interrupt/Resume Functionality Implementation

This document explains how the LinkedIn Career Bot now implements proper interrupt and resume functionality using LangGraph's human-in-the-loop pattern.

## Overview

The bot now follows the exact flow described in the [LangGraph Human-in-the-Loop documentation](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/):

1. **Interrupt**: Graph execution pauses and waits for human input
2. **Resume**: Graph execution continues from where it left off using saved state
3. **State Persistence**: All state is saved using MongoDB checkpointer between interruptions

## Key Changes Made

### 1. Replaced `graph.invoke()` with `graph.stream()`

**Before (Problematic)**:
```python
result = qa_graph.invoke(state, config=config)
```

**After (Correct)**:
```python
for event in qa_graph.stream(state, config=config):
    if event["event"] == "on_chain_end":
        # Handle completion
        break
    elif event["event"] == "on_chain_error":
        # Handle errors
        break
```

### 2. Proper Interrupt Points

Each node now properly interrupts after execution:

```python
def analyze_profile_node(state: AgentState) -> dict:
    # ... process profile analysis ...
    messages.append(AIMessage(res.text))
    state["messages"] = messages
    return interrupt("continue_chat")  # Wait for next human input
```

### 3. State Management

The app now maintains:
- `current_graph_state`: Last saved state from the graph
- `current_graph`: Which graph is currently active
- `waiting_for_input`: Whether the graph is waiting for human input

### 4. Resume Functionality

```python
def resume_graph_from_state(graph, thread_id, user_question, job_description):
    """Resume graph execution from the last saved state"""
    config = {"configurable": {"thread_id": thread_id}}
    last_state = memory.get(config)
    
    if last_state:
        state = last_state.copy()
        state["messages"].append(HumanMessage(content=user_question))
        state["current_job_description"] = job_description
        return state
    return None
```

## Flow Diagram

```
User Input → Resume Graph → Execute Node → Interrupt → Wait for Input
     ↑                                                      ↓
     ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

## Two Main Graphs

### 1. QA Graph (`qa_graph`)
- **Entry**: LinkedIn scraper
- **Flow**: Scraper → Router → Analysis Nodes → **Router** → Interrupt
- **Interrupt Points**: After each analysis node execution, returns to router
- **Resume**: From router to handle next question
- **Routing**: Router understands any prompt and routes to appropriate node (profile analysis, job fit, enhancement, general QA, or career plan)

### 2. Career Plan Graph (`plan_graph`)
- **Entry**: Web search (always done)
- **Flow**: Web Search → Career Plan → Interrupt → Review → Interrupt
- **Interrupt Points**: After career plan generation and after review processing
- **Resume**: From last saved state for feedback processing
- **Web Search**: Always performed to provide current resources and links

## Usage Examples

### Starting a New Session
1. Enter LinkedIn URL
2. Click "Start Assistant"
3. Graph scrapes profile and interrupts
4. Status shows "Waiting for your input"

### Asking Questions
1. Type question in "Ask AI" field
2. Click "Ask AI"
3. Graph resumes from last state
4. Router determines which node to call based on prompt understanding
5. Node executes and returns to router
6. Router interrupts to wait for next question
7. Response appears in chat history

### Career Planning
1. Enter career goal brief
2. Click "Generate Career Plan"
3. Plan graph executes with web search (always done)
4. Graph interrupts after plan generation
5. User can provide feedback
6. Graph resumes to process feedback
7. Graph interrupts again to wait for next input

## State Persistence

All state is automatically saved to MongoDB using the checkpointer:

```python
# In graph compilation
return builder.compile(checkpointer=memory)

# In app usage
config = {"configurable": {"thread_id": st.session_state.thread_id}}
```

## Profile Data Loading

Profile data is automatically loaded from the checkpoint when generating career plans:

```python
# Try to resume from last saved state first
state = resume_graph_from_state(plan_graph, st.session_state.thread_id, f"Create a career plan: {career_brief}", job_description)

if state is None:
    # Load profile data from checkpoint
    checkpoint_state = memory.get({"configurable": {"thread_id": st.session_state.thread_id}})
    if checkpoint_state and "profile_data" in checkpoint_state:
        profile_data = checkpoint_state["profile_data"]
```

## Debugging

Use the debug buttons to troubleshoot:

- **Debug Interrupt/Resume State**: Shows current graph state
- **Debug Current Thread**: Shows thread-specific information
- **Show Thread Statistics**: Shows memory usage statistics

## Testing

Run the test script to verify functionality:

```bash
python test_interrupt_resume.py
```

This tests:
- QA graph interrupt/resume
- QA graph routing to different nodes
- Plan graph interrupt/resume  
- Memory persistence
- State retrieval
- Checkpoint profile loading

## Benefits

1. **No More Re-running**: Graphs resume from where they left off
2. **State Preservation**: All context is maintained between interactions
3. **Human Control**: Users control when to continue execution
4. **Efficient**: No unnecessary re-computation
5. **Scalable**: State persists across sessions using MongoDB
6. **Smart Routing**: Router understands any prompt and routes appropriately
7. **Always Web Search**: Career plans always include current web resources

## Troubleshooting

### Common Issues

1. **Graph not resuming**: Check if `current_graph_state` exists
2. **State lost**: Verify MongoDB connection and checkpointer
3. **Interrupt not working**: Ensure nodes return `interrupt()` calls
4. **Profile not loading**: Check if profile data exists in checkpoint

### Debug Steps

1. Check debug output for state information
2. Verify thread ID consistency
3. Check MongoDB connection status
4. Review interrupt points in node functions
5. Verify profile data is saved in checkpoint

## Future Enhancements

1. **Visual Graph State**: Show current node in execution
2. **Progress Indicators**: Show completion status
3. **State Rollback**: Allow users to go back to previous states
4. **Batch Processing**: Handle multiple questions in sequence

## Conclusion

The implementation now follows LangGraph best practices for human-in-the-loop workflows. The bot maintains state between interactions, provides clear feedback on execution status, allows users to control the flow of conversation naturally, and ensures that profile data is preserved and reused from checkpoints without unnecessary re-scraping.
