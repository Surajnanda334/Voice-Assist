import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SERPER_API_KEY")
print(f"API Key found: {bool(api_key)}")

url = "https://google.serper.dev/search"
payload = json.dumps({
  "q": "latest news about AI",
  "num": 3
})
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}

try:
    print("Sending request to Serper...")
    response = requests.request("POST", url, headers=headers, data=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
