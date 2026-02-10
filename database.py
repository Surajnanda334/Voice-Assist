from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from config import settings

class DatabaseService:
    client: AsyncIOMotorClient = None
    db = None

    def connect(self):
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[settings.DB_NAME]
            print(f"Connected to MongoDB at {settings.MONGO_URI}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    async def log_conversation(self, query: str, response: str, route: str, search_query: str = None):
        if self.db is None: return
        try:
            document = {
                "timestamp": datetime.now(),
                "query": query,
                "response": response,
                "route": route,
                "search_query": search_query
            }
            await self.db.conversations.insert_one(document)
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")

db_service = DatabaseService()
