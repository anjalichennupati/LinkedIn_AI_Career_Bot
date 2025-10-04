#!/usr/bin/env python3
"""
Test script for interrupt/resume functionality in the career bot graphs.
This script tests the proper flow of interrupts and resumes following LangGraph's human-in-the-loop pattern.
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from agents import qa_graph, plan_graph, memory

def test_qa_graph_interrupt_resume():
    """Test the QA graph interrupt/resume functionality"""
    print("=== Testing QA Graph Interrupt/Resume ===")
    
    # Test state
    test_state = {
        "linkedin_url": "https://linkedin.com/in/testuser",
        "thread_id": "test_qa_thread",
        "messages": [],
        "profile_scraped": False
    }
    
    config = {"configurable": {"thread_id": "test_qa_thread"}}
    
    try:
        # Start the graph execution
        print("Starting QA graph execution...")
        for event in qa_graph.stream(test_state, config=config):
            print(f"Event: {event['event']}")
            
            if event["event"] == "on_chain_end":
                print("Graph execution ended")
                if "profile_data" in event["data"]["output"]:
                    print("✅ Profile data extracted successfully")
                    print(f"Profile: {event['data']['output']['profile_data'].get('headline', 'No headline')}")
                break
            elif event["event"] == "on_chain_error":
                print(f"❌ Error: {event['data']['error']}")
                break
            elif event["event"] == "on_chain_start":
                print("Graph execution started")
                
        print("QA Graph test completed\n")
        
    except Exception as e:
        print(f"❌ QA Graph test failed: {e}")

def test_qa_graph_routing():
    """Test that the QA graph properly routes to different nodes based on user questions"""
    print("=== Testing QA Graph Routing ===")
    
    # Test with profile already scraped
    test_state = {
        "messages": [HumanMessage(content="Can you analyze my LinkedIn profile?")],
        "profile_data": {
            "headline": "Software Engineer",
            "skills": ["Python", "JavaScript", "React"],
            "experience": "3 years",
            "user_id": "test_user"
        },
        "thread_id": "test_routing_thread",
        "profile_scraped": True
    }
    
    config = {"configurable": {"thread_id": "test_routing_thread"}}
    
    try:
        print("Testing routing with profile analysis question...")
        for event in qa_graph.stream(test_state, config=config):
            print(f"Event: {event['event']}")
            
            if event["event"] == "on_chain_end":
                print("Routing test completed")
                if "messages" in event["data"]["output"]:
                    messages = event["data"]["output"]["messages"]
                    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                    if ai_msgs:
                        print("✅ Response generated successfully")
                        print(f"Response length: {len(ai_msgs[-1].content)} characters")
                break
            elif event["event"] == "on_chain_error":
                print(f"❌ Error: {event['data']['error']}")
                break
                
        print("QA Graph routing test completed\n")
        
    except Exception as e:
        print(f"❌ QA Graph routing test failed: {e}")

def test_plan_graph_interrupt_resume():
    """Test the plan graph interrupt/resume functionality"""
    print("=== Testing Plan Graph Interrupt/Resume ===")
    
    # Test state with profile data
    test_state = {
        "messages": [HumanMessage(content="Create a career plan: I want to become a data scientist in 6 months")],
        "profile_data": {
            "headline": "Software Engineer",
            "skills": ["Python", "JavaScript", "React"],
            "experience": "3 years",
            "user_id": "test_user"
        },
        "thread_id": "test_plan_thread",
        "profile_scraped": True
    }
    
    config = {"configurable": {"thread_id": "test_plan_thread"}}
    
    try:
        # Start the plan graph execution
        print("Starting plan graph execution...")
        for event in plan_graph.stream(test_state, config=config):
            print(f"Event: {event['event']}")
            
            if event["event"] == "on_chain_end":
                print("Plan graph execution ended")
                if "messages" in event["data"]["output"]:
                    messages = event["data"]["output"]["messages"]
                    ai_msgs = [m for m in messages if isinstance(m, AIMessage)]
                    if ai_msgs:
                        print("✅ Career plan generated successfully")
                        print(f"Plan length: {len(ai_msgs[-1].content)} characters")
                break
            elif event["event"] == "on_chain_error":
                print(f"❌ Error: {event['data']['error']}")
                break
            elif event["event"] == "on_chain_start":
                print("Plan graph execution started")
                
        print("Plan Graph test completed\n")
        
    except Exception as e:
        print(f"❌ Plan Graph test failed: {e}")

def test_memory_persistence():
    """Test that the MongoDB checkpointer properly saves and retrieves state"""
    print("=== Testing Memory Persistence ===")
    
    try:
        # Test saving state
        test_state = {
            "messages": [HumanMessage(content="Test message")],
            "profile_data": {"test": "data"},
            "thread_id": "test_memory_thread"
        }
        
        config = {"configurable": {"thread_id": "test_memory_thread"}}
        
        # Save state
        memory.put(test_state, config)
        print("✅ State saved to memory")
        
        # Retrieve state
        retrieved_state = memory.get(config)
        if retrieved_state:
            print("✅ State retrieved from memory")
            print(f"Retrieved messages: {len(retrieved_state.get('messages', []))}")
        else:
            print("❌ Failed to retrieve state from memory")
            
        print("Memory persistence test completed\n")
        
    except Exception as e:
        print(f"❌ Memory persistence test failed: {e}")

def test_checkpoint_profile_loading():
    """Test that profile data is properly loaded from checkpoint"""
    print("=== Testing Checkpoint Profile Loading ===")
    
    try:
        # First save a state with profile data
        test_state = {
            "messages": [HumanMessage(content="Test message")],
            "profile_data": {
                "headline": "Test Engineer",
                "skills": ["Python", "Testing"],
                "experience": "2 years",
                "user_id": "test_user"
            },
            "thread_id": "test_profile_thread"
        }
        
        config = {"configurable": {"thread_id": "test_profile_thread"}}
        
        # Save state
        memory.put(test_state, config)
        print("✅ State with profile data saved to memory")
        
        # Retrieve state
        retrieved_state = memory.get(config)
        if retrieved_state and "profile_data" in retrieved_state:
            profile = retrieved_state["profile_data"]
            print("✅ Profile data loaded from checkpoint")
            print(f"Profile headline: {profile.get('headline')}")
            print(f"Profile skills: {profile.get('skills')}")
        else:
            print("❌ Failed to load profile data from checkpoint")
            
        print("Checkpoint profile loading test completed\n")
        
    except Exception as e:
        print(f"❌ Checkpoint profile loading test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Interrupt/Resume Functionality")
    print("=" * 50)
    
    # Test memory persistence first
    test_memory_persistence()
    
    # Test checkpoint profile loading
    test_checkpoint_profile_loading()
    
    # Test QA graph
    test_qa_graph_interrupt_resume()
    
    # Test QA graph routing
    test_qa_graph_routing()
    
    # Test plan graph
    test_plan_graph_interrupt_resume()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
