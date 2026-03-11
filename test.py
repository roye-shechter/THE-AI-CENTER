"""
THE AI CENTER - ENTERPRISE MULTI-AGENT ARCHITECTURE
Author: Roye Schechter
Description: A robust infrastructure for managing multiple AI providers.
Designed for scalability, allowing integration of various LLMs through a unified gateway.
"""

import os
import uuid
import sqlite3
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Response, Request, BackgroundTasks
from google import genai
from google.genai import types
from groq import Groq
from twilio.rest import Client

# --- 1. CONFIGURATION ---
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_FILE = "nexus_system.db"
MEDIA_DIR = "temp_media"
if not os.path.exists(MEDIA_DIR): os.makedirs(MEDIA_DIR)

# --- 2. PERSISTENCE LAYER ---
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (phone_number TEXT PRIMARY KEY, active_model TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, phone_number TEXT, role TEXT, message_text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
init_db()

def get_user_state(phone: str) -> str:
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute("SELECT active_model FROM users WHERE phone_number = ?", (phone,)).fetchone()
        return res[0] if res else "0"

def set_user_state(phone: str, model_id: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO users (phone_number, active_model) VALUES (?, ?) ON CONFLICT(phone_number) DO UPDATE SET active_model = excluded.active_model", (phone, model_id))

# --- 3. COMMUNICATION GATEWAY ---
def send_whatsapp(to_number: str, content: str):
    """Encapsulates Twilio messaging logic."""
    try:
        twilio = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        twilio.messages.create(from_="whatsapp:+14155238886", body=content, to=to_number)
    except Exception as e: print(f"⚠️ Communication Error: {e}")

# --- 4. SEMANTIC ROUTING ENGINE (The Brain) ---
def intent_classifier(user_input: str, has_media: bool) -> str:
    """Classifies user intent to route between high-reasoning and fast-response agents."""
    if has_media or len(user_input) > 280: return "GEMINI"
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "Route to 'GEMINI' for complex/multimodal tasks, 'LLAMA' for chat. Respond ONLY with the tag."},
                      {"role": "user", "content": user_input}],
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=5
        )
        return completion.choices[0].message.content.strip().upper()
    except: return "GEMINI"

# --- 5. MODULAR WORKER ---
def background_worker(From: str, Body: str, NumMedia: str, MediaUrl0: str, MediaContentType0: str):
    user_msg = Body.strip().lower()

    # --- DYNAMIC UI / MENU ---
    if user_msg in ["menu", "תפריט", "היי", "שלום"]:
        menu_msg = (
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀ **AI CENTER**\n\n"
            "אנא בחר את סוכן ה-AI המבוקש:\n\n"
            "0️⃣ **המנהל החכם (Smart Manager)** 🧠\n"
            "1️⃣ ChatGPT (OpenAI)\n"
            "2️⃣ Claude (Anthropic)\n"
            "3️⃣ Gemini (Google)\n"
            "4️⃣ Llama (Meta)\n"
            "5️⃣ Grok (xAI)\n\n"
            "⚠️ **אזהרה:** חלק מהמודלים דורשים חשבון Premium לתשתית החברה המפתחת.\n\n"
            "📍 שלח מספר לבחירה או 'נקה' לאיפוס."
        )
        send_whatsapp(From, menu_msg)
        return

    # User Selection Handler
    if user_msg in ["0", "1", "2", "3", "4", "5"]:
        set_user_state(From, user_msg)
        names = {"0":"המנהל החכם", "1":"ChatGPT", "2":"Claude", "3":"Gemini", "4":"Llama", "5":"Grok"}
        send_whatsapp(From, f"✅ המערכת הוגדרה לשימוש ב: *{names[user_msg]}*")
        return

    # --- EXECUTION ENGINE ---
    current_mode = get_user_state(From)
    is_media = int(NumMedia) > 0

    # OPTION 0: Autonomous Routing Logic
    if current_mode == "0":
        route = intent_classifier(Body, is_media)
        if route == "GEMINI":
            try:
                user_parts = []
                if is_media:
                    res = requests.get(MediaUrl0)
                    path = os.path.join(MEDIA_DIR, str(uuid.uuid4()))
                    with open(path, "wb") as f: f.write(res.content)
                    uploaded = gemini_client.files.upload(file=path)
                    user_parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
                    os.remove(path)

                    # Logic for Voice / Vision
                    instr = "Summarize this audio: Priority, Content, Suggested Action." if "audio" in (MediaContentType0 or "") else Body or "Analyze this content."
                else:
                    instr = Body

                user_parts.append(types.Part.from_text(text=instr))
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config={'system_instruction': "You are Roye's Smart AI Manager. Hebrew only.", 'tools': [{'google_search': {}}]},
                    contents=[types.Content(role="user", parts=user_parts)]
                )
                answer = response.text
            except: answer = "שגיאה זמנית בחיבור לסוכן הניהול."
        else:
            # Fallback to Llama (Fast Chat)
            try:
                comp = groq_client.chat.completions.create(messages=[{"role":"user", "content":Body}], model="llama-3.3-70b-versatile")
                answer = comp.choices[0].message.content
            except: answer = "שגיאה בחיבור לסוכן המהיר."

        send_whatsapp(From, answer)
        return

    # OPTIONS 1-5: Manual Infrastructure
    # Note: These serve as placeholders. When API keys for OpenAI/Anthropic/xAI are added,
    # the logic below will dispatch to their respective SDKs.
    send_whatsapp(From, f"🤖 סוכן {current_mode} ממתין להגדרת מפתח API חיצוני. כרגע מופעל במצב תשתית בלבד.")

# --- 6. ENTRY POINT ---
app = FastAPI()

@app.post("/webhook")
async def whatsapp_webhook(background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(""),
                           NumMedia: str = Form("0"), MediaUrl0: str = Form(None), MediaContentType0: str = Form(None)):
    background_tasks.add_task(background_worker, From, Body, NumMedia, MediaUrl0, MediaContentType0)
    return Response(content="<Response></Response>", media_type="text/xml")