import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_CONFIG = {
    "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
    "database_name": os.getenv("MONGODB_DB_NAME", "career_bot"),
    "collections": {
        "procedural_memories": "procedural_memories",
        "checkpoints": "checkpoints",
        "websearch": "websearch",
        "user_profiles": "user_profiles"
    },
    "connection_timeout": int(os.getenv("MONGODB_TIMEOUT", "5000")),
    "max_pool_size": int(os.getenv("MONGODB_MAX_POOL_SIZE", "50")),
    "retry_writes": os.getenv("MONGODB_RETRY_WRITES", "true").lower() == "true"
}

# Procedural Memory Configuration
PROCEDURAL_MEMORY_CONFIG = {
    "max_entries_per_user": int(os.getenv("PROCEDURAL_MAX_ENTRIES", "30")),
    "similarity_threshold": float(os.getenv("PROCEDURAL_SIMILARITY_THRESHOLD", "0.7")),
    "max_procedure_length": int(os.getenv("PROCEDURAL_MAX_LENGTH", "2500")),
    "max_query_length": int(os.getenv("PROCEDURAL_MAX_QUERY_LENGTH", "600")),
    "default_top_k": int(os.getenv("PROCEDURAL_DEFAULT_TOP_K", "3"))
}
