# agents/memory_manager.py
import pickle
import os

MEMORY_FILE = "session_memory.pkl"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def update_memory(memory):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(memory, f)
