import json
import httpx
from config import settings

class SearchService:
    def __init__(self):
        self.api_key = settings.SERPER_API_KEY
        self.url = "https://google.serper.dev/search"

    async def search(self, query: str):
        if not self.api_key: return "Error: SERPER_API_KEY not found."
        
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        payload = json.dumps({"q": query})
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=headers, data=payload)
                if response.status_code == 200:
                    results = response.json()
                    snippets = []
                    if "organic" in results:
                        for item in results["organic"][:3]:
                            snippets.append(f"{item.get('title')}: {item.get('snippet')}")
                    return "\n".join(snippets) if snippets else "No results found."
                else:
                    return f"Search Error: {response.status_code}"
        except Exception as e:
            return f"Search Exception: {e}"

search_service = SearchService()
