"""
🌐 ROYE SHECHTER's Universal AI Gateway
By: Roye Schechter
An advanced multi-model gateway connecting the world's best AI engines.
"""

import os
import uuid
import sqlite3
import asyncio
import requests
from contextlib import asynccontextmanager

# Third-party integrations
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Response, Request
from fastapi.staticfiles import StaticFiles
from twilio.twiml.messaging_response import MessagingResponse
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic

# --- Environment Setup ---
load_dotenv()

# Secure Credential Retrieval
API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# --- Model & File Config ---
MODEL_NAME_CHAT = 'gemini-2.0-flash'
MODEL_NAME_IMAGE = 'imagen-4.0-generate-001'
MODEL_NAME_LLAMA = "llama-3.3-70b-versatile"
DB_FILE = "bot_database.db"
IMAGES_DIR = "images"

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# --- Clients Initialization ---
client = genai.Client(api_key=API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
grok_client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# --- Database Helper Functions ---

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS users (phone_number TEXT PRIMARY KEY, active_model TEXT)')
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, phone_number TEXT, role TEXT, message_text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

init_db()

def get_user_model(phone: str):
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute("SELECT active_model FROM users WHERE phone_number = ?", (phone,)).fetchone()
        return res[0] if res else None

def set_user_model(phone: str, model_id: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO users (phone_number, active_model) VALUES (?, ?) ON CONFLICT(phone_number) DO UPDATE SET active_model = excluded.active_model", (phone, model_id))
        conn.commit()

def clear_chat_history(phone: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("DELETE FROM chat_history WHERE phone_number = ?", (phone,))
        conn.commit()

def save_message(phone: str, role: str, text: str):
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO chat_history (phone_number, role, message_text) VALUES (?, ?, ?)", (phone, role, text))
        conn.commit()

def get_chat_history_gemini(phone: str):
    """ FIX: Fetches and formats history for Gemini API """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, message_text FROM chat_history WHERE phone_number = ? ORDER BY id ASC", (phone,))
        return [types.Content(role=r, parts=[types.Part.from_text(text=t)]) for r, t in cursor.fetchall()]

# --- Background Task: Database Cleanup ---
async def cleanup_old_sessions():
    while True:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE timestamp <= datetime('now', '-24 hours')")
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Cleanup: {e}", flush=True)
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_old_sessions())
    yield
    cleanup_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# --- Main Webhook Interface ---

@app.post("/webhook")
async def bot(request: Request, From: str = Form(...), Body: str = Form(""), NumMedia: str = Form("0"), MediaUrl0: str = Form(None), MediaContentType0: str = Form(None)):
    twiml = MessagingResponse()
    user_msg = Body.strip().lower()

    # 1. GLOBAL COMMANDS: Menu & Context Reset
    if user_msg in ["תפריט", "menu", "היי", "שלום", "hi", "hello"]:
        menu_text = (
            "⠀⠀⠀⠀⠀⠀⠀⠀⠀ *AI CENTER*\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "שלום. אנא בחר מודל שפה להמשך השיחה:\n\n"
            "1️⃣ *ChatGPT 4o* (OpenAI)\n"
            "2️⃣ *Claude 3.5* (Anthropic)\n"
            "3️⃣ *Grok-2* (xAI)\n"
            "4️⃣ *Gemini 2.0* (Google)\n"
            "5️⃣ *Llama 3.3* (Meta)\n\n"
            "⚠️ ייתכן וחלק מהכלים סגורים למנויי Premium בלבד.\n\n"
            "🧹 לאיפוס היסטוריית השיחה, הקלד: *נקה*\n\n"
            "⚡ _Created by Roye Schechter_"
        )
        twiml.message(menu_text)
        return Response(content=str(twiml).encode('utf-8'), media_type="text/xml")

    if user_msg in ["נקה", "חדש", "clear", "reset"]:
        clear_chat_history(From)
        twiml.message("היסטוריית השיחה נוקתה בהצלחה! 🧹")
        return Response(content=str(twiml).encode('utf-8'), media_type="text/xml")

    # 2. MODEL SELECTION
    if user_msg in ["1", "2", "3", "4", "5"]:
        current_model = get_user_model(From)
        if current_model != user_msg:
            clear_chat_history(From)
        set_user_model(From, user_msg)

        if user_msg == "4":
            msg = (
                "✅ חיבור למודל Google Gemini 2.0 הושלם.\n\n"
                "יכולות המערכת כעת:\n"
                "- שיחת טקסט\n"
                "- ניתוח הודעות קוליות\n"
                "- ניתוח תמונות מצורפות\n"
                "- יצירת תמונות (התחל הודעה במילה 'צייר')\n\n"
                "המערכת ממתינה להודעתך."
                )
        elif user_msg == "5":
            msg = (
                "✅ חיבור למודל Meta Llama 3.3 הושלם.\n\n"
                "יכולות המערכת כעת:\n"
                "- שיחת טקסט בלבד (מודל בעיבוד מהיר)\n\n"
                "המערכת ממתינה להודעתך."
            )
        else:
            msg = (
                "💎 *שירותי Premium*\n"
                f"בחרת במודל מספר {user_msg}.\n"
                "המערכת מזהה כי נדרש שדרוג VIP לגישה זו. 💳\n\n"
                "🙌 לחזרה למודלים החינמיים, שלח 'תפריט'."
            )

        twiml.message(msg)
        return Response(content=str(twiml).encode('utf-8'), media_type="text/xml")

    # 3. INTERACTION LOGIC
    current_model = get_user_model(From)
    if not current_model:
        twiml.message("ברוך הבא למרכז ה-AI! שלח 'תפריט' כדי להתחיל.")
        return Response(content=str(twiml).encode('utf-8'), media_type="text/xml")

    if current_model in ["1", "2", "3"]:
        twiml.message("⚠️ הגישה למודל זה מוגבלת למנויי פרימיום. שלח 'תפריט' למעבר ל-Gemini או Llama.")

    elif current_model == "4":
        # Image Generation
        if user_msg.startswith("צייר") or user_msg.startswith("draw"):
            try:
                prompt_prefix = "צייר" if user_msg.startswith("צייר") else "draw"
                image_prompt = Body[len(prompt_prefix):].strip()
                translated = f"Generate a high-quality professional image of: {image_prompt}"
                response_image = client.models.generate_images(model=MODEL_NAME_IMAGE, prompt=translated)

                if response_image and response_image.generated_images:
                    raw_bytes = response_image.generated_images[0].image.image_bytes
                    filename = f"{uuid.uuid4()}.png"
                    with open(os.path.join(IMAGES_DIR, filename), "wb") as f: f.write(raw_bytes)

                    base_url = str(request.base_url)
                    public_image_url = f"{base_url if base_url.endswith('/') else base_url + '/'}images/{filename}"
                    msg = twiml.message("✨ הנה היצירה המבוקשת:")
                    msg.media(public_image_url)
                    save_message(From, "user", f"[DRAW] {image_prompt}")
                    save_message(From, "model", f"[IMAGE] {filename}")
            except Exception as e:
                twiml.message("יצירת תמונות דורשת מנוי פרימיום פעיל. 💳")
        # Multimodal Chat
        else:
            try:
                history = get_chat_history_gemini(From)
                prompt_text = Body.strip() or "נתח את המדיה."
                media_parts = []
                if int(NumMedia) > 0 and MediaUrl0:
                    m_res = requests.get(MediaUrl0, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
                    if m_res.status_code == 200:
                        mime = MediaContentType0.split(';')[0].strip()
                        media_parts.append(types.Part.from_bytes(data=m_res.content, mime_type=mime))

                user_parts = [types.Part.from_text(text=prompt_text)] + media_parts
                history.append(types.Content(role="user", parts=user_parts))

                res = client.models.generate_content(model=MODEL_NAME_CHAT, config={'system_instruction': "ענה תמיד בעברית קצרה ולעניין."}, contents=history)
                save_message(From, "user", prompt_text + (" [Media]" if media_parts else ""))
                save_message(From, "model", res.text)
                twiml.message(res.text)
            except Exception as e:
                twiml.message("חלה תקלה קטנה, המערכת מאתחלת את עצמה... נסה שוב.")

    elif current_model == "5":
        try:
            msgs = [{"role": "system", "content": "ענה תמיד בעברית קצרה ולעניין."}]
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("SELECT role, message_text FROM chat_history WHERE phone_number = ? ORDER BY id ASC", (From,)).fetchall()
                for r, t in rows: msgs.append({"role": "assistant" if r == "model" else "user", "content": t})

            msgs.append({"role": "user", "content": Body.strip() or "שלום"})
            comp = groq_client.chat.completions.create(messages=msgs, model=MODEL_NAME_LLAMA)
            ans = comp.choices[0].message.content
            save_message(From, "user", Body.strip())
            save_message(From, "model", ans)
            twiml.message(ans)
        except Exception as e:
            twiml.message("שרתי ה-AI עמוסים כרגע, נסה שוב בעוד רגע.")

    return Response(content=str(twiml).encode('utf-8'), media_type="text/xml")

@app.get("/")
def health(): return {"status": "AI HUB Online", "owner": "Roye Schechter"}