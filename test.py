from pymongo import MongoClient
import os

# same URI you put in your .env
uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    client.server_info()  # force connection
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ Not connected:", e)
