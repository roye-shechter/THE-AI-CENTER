"""
THE AI CENTER - ENTERPRISE MULTI-AGENT ARCHITECTURE
Author: Roye Schechter
Description: Robust infrastructure managing multiple AI providers + Long-Term Memory.
"""

import os
import uuid
import sqlite3
import requests
import io
import time
import mimetypes
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Response, Request, BackgroundTasks
from PyPDF2 import PdfReader
from pinecone import Pinecone
from google import genai
from google.genai import types
from groq import Groq
from twilio.rest import Client

# ==========================================
# MODULE 1: CONFIGURATION & INITIALIZATION
# ==========================================
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Pinecone Vector Database for Long-Term Memory
pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME") or "nexus-knowledge")

DB_FILE = "nexus_system.db"
MEDIA_DIR = "temp_media"
if not os.path.exists(MEDIA_DIR): os.makedirs(MEDIA_DIR)

# ==========================================
# MODULE 2: PERSISTENCE LAYER (DATABASE)
# ==========================================
def init_db():
    """Initializes the SQLite database for user states and history."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (phone_number TEXT PRIMARY KEY, active_model TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, phone_number TEXT, role TEXT, message_text TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)')
init_db()

def get_user_state(phone: str) -> str:
    """Retrieves the currently selected AI agent for the user."""
    with sqlite3.connect(DB_FILE) as conn:
        res = conn.execute("SELECT active_model FROM users WHERE phone_number = ?", (phone,)).fetchone()
        return res[0] if res else "0"

def set_user_state(phone: str, model_id: str):
    """Updates the user's selected AI agent."""
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("INSERT INTO users (phone_number, active_model) VALUES (?, ?) ON CONFLICT(phone_number) DO UPDATE SET active_model = excluded.active_model", (phone, model_id))

# ==========================================
# MODULE 3: COMMUNICATION GATEWAY
# ==========================================
def send_whatsapp(to_number: str, content: str, media_url: str = None):
    """
    Sends WhatsApp messages via Twilio.
    Supports sending media (images) via public URLs.
    """
    try:
        twilio = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        limit = 1500
        parts = [content[i:i + limit] for i in range(0, len(content), limit)] if content else [""]

        for i, part in enumerate(parts):
            kwargs = {"from_": "whatsapp:+14155238886", "to": to_number, "body": part}

            # Attach media ONLY to the first message part
            if media_url and i == 0:
                kwargs["media_url"] = [media_url]

            twilio.messages.create(**kwargs)
            time.sleep(0.5)
    except Exception as e:
        print(f"🔥 Twilio Error: {e}")

# ==========================================
# MODULE 4: MEMORY ENGINE (RAG)
# ==========================================
def get_embedding(text: str):
    """Generates vector embeddings for text chunks using Gemini."""
    target_model = "gemini-embedding-001"
    try:
        res = gemini_client.models.embed_content(
            model=target_model,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
        )
        return res.embeddings[0].values
    except Exception as e:
        res = gemini_client.models.embed_content(
            model=f"models/{target_model}",
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
        )
        return res.embeddings[0].values

def ingest_pdf(file_url: str, phone: str):
    """Downloads a PDF, extracts text, chunks it, and uploads vectors to Pinecone."""
    try:
        res = requests.get(file_url, auth=(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")))
        if res.status_code != 200: return False
        reader = PdfReader(io.BytesIO(res.content))
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text.strip(): return False

        chunks = [text[i:i + 1000] for i in range(0, len(text), 800)]
        vectors = []
        for chunk in chunks:
            if len(chunk.strip()) < 50: continue
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": get_embedding(chunk),
                "metadata": {"text": chunk, "phone": phone}
            })
        index.upsert(vectors=vectors, namespace="nexus-pro-docs")
        return True
    except Exception as e:
        print(f"Error in document ingestion: {e}")
        return False

def retrieve_context(query: str, phone: str):
    """Searches Pinecone for relevant document chunks based on the user's query."""
    res = index.query(
        vector=get_embedding(query), top_k=3, include_metadata=True,
        namespace="nexus-pro-docs", filter={"phone": {"$eq": phone}}
    )
    return "\n---\n".join([m.metadata['text'] for m in res.matches if m.score > 0.7])

# ==========================================
# MODULE 5: SEMANTIC ROUTING ENGINE
# ==========================================
def intent_classifier(user_input: str, has_media: bool) -> str:
    """
    Decides whether to route the request to Gemini (for media/complex tasks)
    or to Llama (for fast text-based chat).
    """
    if has_media or len(user_input) > 280: return "GEMINI"
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": "Route to 'GEMINI' for complex/multimodal tasks, 'LLAMA' for chat. Respond ONLY with the tag."},
                      {"role": "user", "content": user_input}],
            model="llama-3.3-70b-versatile", temperature=0, max_tokens=5
        )
        return completion.choices[0].message.content.strip().upper()
    except: return "GEMINI"

# ==========================================
# MODULE 6: CORE WORKER (LOGIC CONTROLLER)
# ==========================================
def background_worker(From: str, Body: str, NumMedia: str, MediaUrl0: str, MediaContentType0: str, base_url: str = ""):
    """Main function handling incoming WhatsApp messages in the background."""
    user_msg = Body.strip().lower()
    is_media = int(NumMedia) > 0
    content_type = (MediaContentType0 or "").lower()

    # --- 6.1 DYNAMIC UI / MENU DISPLAY ---
    if user_msg in ["menu", "תפריט", "היי", "שלום", "נקה", "hi", "hey", "אהלן"] and not is_media:
        menu_msg = (
            "\u202B🔹 ══════ *AI CENTER* ══════ 🔹\u202C\n\n"
            "\u202Bאנא בחר את סוכן ה-AI המבוקש:\u202C\n\n"
            "\u200E0️⃣ *המנהל החכם (Smart Manager)* 🧠\n\n"
            "\u200E1️⃣ *ChatGPT (OpenAI)*\n\n"
            "\u200E2️⃣ *Claude (Anthropic)*\n\n"
            "\u200E3️⃣ *Gemini (Google)*\n\n"
            "\u200E4️⃣ *Llama (Meta)*\n\n"
            "\u200E5️⃣ *Grok (xAI)*\n\n"
            "\u202B⚠️ *אזהרה:* חלק מהמודלים דורשים חשבון Premium לתשתית החברה המפתחת.\u202C\n\n"
            "\u202B📍 שלח מספר לבחירה או 'נקה' לאיפוס.\u202C\n\n"
            "\u202BDeveloped by Roye Schechter ⚡\u202C"
        )
        send_whatsapp(From, menu_msg)
        return

    # --- 6.2 USER SELECTION HANDLER ---
    if user_msg in ["0", "1", "2", "3", "4", "5"]:
        set_user_state(From, user_msg)
        if user_msg == "0":
            msg = (
                "\u202B✅ *המנהל החכם הופעל (Smart Manager)* 🧠\u202C\n\n"
                "\u202Bתכונות המערכת:\u202C\n\n"
                "\u202B📄 *ניהול ידע:* למידה, ניתוח ושימור תוכן מקבצי PDF.\u202C\n\n\n"
                "\u202B⚡ *אופטימיזציה:* ניתוב אוטומטי למודל המהיר ביותר.\u202C\n\n\n"
                "\u202B🎨 *יצירה וניתוח ויזואלי:* יצירת תמונות חדשות (התחל משפט במילה 'צייר') וניתוח תמונות שתשלח.\u202C\n\n\n"
                "\u202B🎤 *תמלול וניתוח שמע:* שלח הקלטת קול (Voice Note) או קובץ שמע, והבוט יתמלל וינתח אותם.\u202C\n\n\n"
            )
            send_whatsapp(From, msg)
        else:
            names = {"1": "ChatGPT", "2": "Claude", "3": "Gemini", "4": "Llama", "5": "Grok"}
            send_whatsapp(From, f"✅ המערכת הוגדרה לשימוש ב: *{names[user_msg]}*")
        return

    # --- 6.3 EXECUTION ENGINE ---
    current_mode = get_user_state(From)

    # OPTION 0: Smart Manager
    if current_mode == "0":

        # ---> תכונת הציור: בדיקה אם המשתמש ביקש תמונה <---
        if Body.startswith("צייר"):
            send_whatsapp(From, "🎨 *מצייר את זה עבורך, רק כמה שניות...*")
            try:
                prompt = Body.replace("צייר", "").strip()
                res = gemini_client.models.generate_images(
                    model='imagen-3.0-generate-001',
                    prompt=prompt,
                    config=types.GenerateImagesConfig(number_of_images=1, output_mime_type="image/jpeg")
                )
                filename = f"{uuid.uuid4()}.jpg"
                filepath = os.path.join(MEDIA_DIR, filename)

                # שמירת התמונה בשרת הזמני
                with open(filepath, "wb") as f:
                    f.write(res.generated_images[0].image.image_bytes)

                # שליחת הקישור של התמונה ל-Twilio
                media_link = f"{base_url}/media/{filename}"
                send_whatsapp(From, f"✅ הנה הציור שלך עבור:\n_{prompt}_", media_url=media_link)
            except Exception as e:
                print(f"🔥 Image Gen Error: {e}")
                send_whatsapp(From, "❌ *שגיאה ביצירת התמונה.* ייתכן שהתיאור חסום (Safety) או שהשרת עמוס.")
            return

        # Handle PDF uploads
        if is_media and "pdf" in content_type:
            send_whatsapp(From, "📚 *לומד את המסמך...*")
            success = ingest_pdf(MediaUrl0, From)
            if success: send_whatsapp(From, "✅ *סיימתי!*")
            if not Body: return
            time.sleep(1)

        # Retrieve context from Pinecone
        context = retrieve_context(Body, From) if Body else ""
        system_prompt = "You are the user assistant. ALWAYS reply in Hebrew. Be concise and professional. Do not exceed 500 characters unless necessary."
        if context: system_prompt += f"\n\nContext from documents:\n{context}"

        # Route dynamically
        route = intent_classifier(Body, is_media)

        if route == "GEMINI":
            try:
                user_parts = [types.Part.from_text(text=Body or "Analyze this file")]
                if is_media and "pdf" not in content_type:
                    res = requests.get(MediaUrl0,
                                       auth=(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")))
                    ext = mimetypes.guess_extension(content_type) or ""
                    path = os.path.join(MEDIA_DIR, f"{uuid.uuid4()}{ext}")
                    with open(path, "wb") as f: f.write(res.content)
                    uploaded = gemini_client.files.upload(file=path, config={'mime_type': content_type})
                    user_parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=content_type))
                    os.remove(path)

                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[{"google_search": {}}]
                    ),
                    contents=[types.Content(role="user", parts=user_parts)]
                )
                answer = response.text
            except Exception as e:
                print(f"🔥 Agent 0 (gemini) Error: {e}")
                answer = "❌ שגיאה זמנית בחיבור לסוכן 0."
        else:
            try:
                comp = groq_client.chat.completions.create(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": Body}],
                    model="llama-3.3-70b-versatile"
                )
                answer = comp.choices[0].message.content
            except Exception as e:
                print(f"🔥 Agent 0 (Llama) Error: {e}")
                answer = "❌ שגיאה בחיבור לסוכן המהיר."

        send_whatsapp(From, answer)
        return

    # OPTIONS 1-5: DIRECT AI AGENTS
    elif current_mode in ["1", "2", "3", "4", "5"]:
        answer = ""
        short_body = f"{Body} (Reply concisely in Hebrew)"

        if current_mode == "1":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                comp = client.chat.completions.create(model="gpt-4o",
                                                      messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except:
                answer = "⚠️ *החיבור ל-ChatGPT נכשל.*"
        elif current_mode == "2":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024,
                                             messages=[{"role": "user", "content": short_body}])
                answer = msg.content[0].text
            except:
                answer = "⚠️ *החיבור ל-Claude נכשל.*"
        elif current_mode == "3":
            try:
                # תוקן כאן מ- gemini-2.5-flash ל- gemini-2.0-flash
                res = gemini_client.models.generate_content(model="gemini-2.0-flash-exp", contents=[short_body])
                answer = res.text
            except:
                answer = "❌ חיבור ל-Gemini נכשל."
        elif current_mode == "4":
            try:
                comp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile",
                                                           messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except:
                answer = "❌ חיבור ל-Llama נכשל."
        elif current_mode == "5":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
                comp = client.chat.completions.create(model="grok-beta",
                                                      messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except:
                answer = "⚠️ *החיבור ל-Grok נכשל.*"

        if answer: send_whatsapp(From, answer)
        return

# ==========================================
# MODULE 7: ENTRY POINT (FASTAPI)
# ==========================================
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the media directory so Twilio can download generated images
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

@app.post("/webhook")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks, From: str = Form(...),
                           Body: str = Form(""),
                           NumMedia: str = Form("0"), MediaUrl0: str = Form(None), MediaContentType0: str = Form(None)):
    """Receives webhook requests from Twilio and passes them to the background worker."""
    # Capture the exact base URL of the Render server dynamically
    base_url = str(request.base_url).rstrip("/")

    background_tasks.add_task(background_worker, From, Body, NumMedia, MediaUrl0, MediaContentType0, base_url)
    return Response(content="<Response></Response>", media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)