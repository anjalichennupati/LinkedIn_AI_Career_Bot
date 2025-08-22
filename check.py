from pymongo import MongoClient
import os

uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(uri)

db = client["langgraph"]
coll = db["checkpoints"]

print("Collections in langgraph:", db.list_collection_names())
print("Number of checkpoints:", coll.count_documents({}))

for doc in coll.find().limit(3):
    print(doc)
