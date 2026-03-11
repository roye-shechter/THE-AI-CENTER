import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("📡 Fetching all model names available for your API Key:")
try:
    # פשוט מדפיסים את כל השמות שיש ברשימה
    for m in client.models.list():
        print(f"-> {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")