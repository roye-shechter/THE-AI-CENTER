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
def send_whatsapp(to_number: str, content: str):
    """
    Sends WhatsApp messages via Twilio.
    Includes logic to split messages exceeding Twilio's 1600-character limit.
    """
    try:
        twilio = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

        # Limit set to 1500 to provide a safe buffer below the 1600 threshold
        limit = 1500
        parts = [content[i:i + limit] for i in range(0, len(content), limit)]

        for part in parts:
            twilio.messages.create(
                from_="whatsapp:+14155238886",  # Verify this is the new active sandbox number
                body=part,
                to=to_number
            )
            time.sleep(0.5)  # Pause to maintain message sequence
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
def background_worker(From: str, Body: str, NumMedia: str, MediaUrl0: str, MediaContentType0: str):
    """Main function handling incoming WhatsApp messages in the background."""
    user_msg = Body.strip().lower()
    is_media = int(NumMedia) > 0
    content_type = (MediaContentType0 or "").lower()

    # --- 6.1 DYNAMIC UI / MENU DISPLAY ---
    if user_msg in ["menu", "תפריט", "היי", "שלום", "נקה", "hi", "hey", "אהלן"] and not is_media:
        menu_msg = (
            "\u200E             *AI CENTER*\n\n"
            "\u200Eאנא בחר את סוכן ה-AI המבוקש:\n\n"
            "\u200E0️⃣ *המנהל החכם (Smart Manager)* 🧠\n\n"
            "\u200E1️⃣ *ChatGPT (OpenAI)*\n\n"
            "\u200E2️⃣ *Claude (Anthropic)*\n\n"
            "\u200E3️⃣ *Gemini (Google)*\n\n"
            "\u200E4️⃣ *Llama (Meta)*\n\n"
            "\u200E5️⃣ *Grok (xAI)*\n\n"
            "\u202B⚠️ *אזהרה:* חלק מהמודלים דורשים חשבון Premium לתשתית החברה המפתחת.\u202C\n\n"
            "\u202B📍 שלח מספר לבחירה או 'נקה' לאיפוס.\u202C\n\n"
            "\u202B_Developed by Roye Schechter_ ⚡\u202C"
        )
        send_whatsapp(From, menu_msg)
        return

    # --- 6.2 USER SELECTION HANDLER ---
    # Handles state changes when the user selects a number from the menu
    if user_msg in ["0", "1", "2", "3", "4", "5"]:
        set_user_state(From, user_msg)
        if user_msg == "0":
            msg = (
                "\u202B✅ *המנהל החכם הופעל (Smart Manager)* 🧠\u202C\n\n"
                "\u202B*תכונות המערכת:*\u202C\n\n"
                "\u202B📄 *ניהול ידע:* למידה, ניתוח ושימור תוכן מקבצי PDF בזיכרון המערכת.\u202C\n\n\n"
                "\u202B⚡ *אופטימיזציה:* התאמה אוטומטית של כלי ה-AI המהיר והיעיל ביותר לשאלה.\u202C\n\n\n"
                "\u202B🎤 *עיבוד רב-מודלי:* ניתוח תמונות, הקלטות קוליות וקבצים ויזואליים.\u202C\n\n\n"
                "\u202B_Developed by Roye Schechter_ ⚡\u202C"
            )
            send_whatsapp(From, msg)
        else:
            names = {"1":"ChatGPT", "2":"Claude", "3":"Gemini", "4":"Llama", "5":"Grok"}
            send_whatsapp(From, f"✅ המערכת הוגדרה לשימוש ב: *{names[user_msg]}*")
        return

    # --- 6.3 EXECUTION ENGINE ---
    # Executes the logic based on the user's currently selected agent
    current_mode = get_user_state(From)

    # OPTION 0: Smart Manager (Routing + Memory RAG)
    if current_mode == "0":
        # Handle PDF uploads
        if is_media and "pdf" in content_type:
            send_whatsapp(From, "📚 *לומד את המסמך...*")
            success = ingest_pdf(MediaUrl0, From)
            if success: send_whatsapp(From, "✅ *סיימתי!*")
            if not Body: return
            time.sleep(1)

        # Retrieve context from Pinecone
        context = retrieve_context(Body, From) if Body else ""
        system_prompt = "You are Roye's assistant. ALWAYS reply in Hebrew. Be concise and professional. Do not exceed 500 characters unless necessary."
        if context: system_prompt += f"\n\nContext from documents:\n{context}"

        # Route dynamically
        route = intent_classifier(Body, is_media)

        if route == "GEMINI":
            try:
                user_parts = [types.Part.from_text(text=Body or "Analyze this file")]
                # Handle images/audio for Agent 0
                if is_media and "pdf" not in content_type:
                    res = requests.get(MediaUrl0, auth=(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")))
                    ext = mimetypes.guess_extension(content_type) or ""
                    path = os.path.join(MEDIA_DIR, f"{uuid.uuid4()}{ext}")
                    with open(path, "wb") as f: f.write(res.content)
                    uploaded = gemini_client.files.upload(file=path, config={'mime_type': content_type})
                    user_parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=content_type))
                    os.remove(path) # Clean up temp files

                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    config={'system_instruction': system_prompt},
                    contents=[types.Content(role="user", parts=user_parts)]
                )
                answer = response.text
            except Exception as e:
                print(f"🔥 Agent 0 (Gemini) Error: {e}")
                answer = "❌ שגיאה זמנית בחיבור לסוכן 0."
        else:
            # Fallback to Groq Llama for fast text
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

    # OPTIONS 1-5: DIRECT AI AGENTS (Bypass Routing and Memory)
    elif current_mode in ["1", "2", "3", "4", "5"]:
        answer = ""
        short_body = f"{Body} (Reply concisely in Hebrew)"

        # 1. ChatGPT (OpenAI)
        if current_mode == "1":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                comp = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except Exception as e:
                answer = "⚠️ *החיבור ל-ChatGPT נכשל.*\nהאם הגדרת מפתח API חוקי והזנת כרטיס אשראי באתר של OpenAI?"

        # 2. Claude (Anthropic)
        elif current_mode == "2":
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                msg = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[{"role": "user", "content": short_body}])
                answer = msg.content[0].text
            except Exception as e:
                answer = "⚠️ *החיבור ל-Claude נכשל.*\nהאם הגדרת מפתח API חוקי והזנת כרטיס אשראי באתר של Anthropic?"

        # 3. Gemini (Direct without Memory)
        elif current_mode == "3":
            try:
                res = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=[short_body])
                answer = res.text
            except Exception as e:
                print(f"🔥 Agent 3 Error: {e}")
                answer = "❌ חיבור ישיר ל-Gemini נכשל."

        # 4. Llama (Direct via Groq)
        elif current_mode == "4":
            try:
                comp = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except Exception as e:
                print(f"🔥 Agent 4 Error: {e}")
                answer = "❌ חיבור ל-Llama נכשל."

        # 5. Grok (xAI)
        elif current_mode == "5":
            try:
                import openai
                client = openai.OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")
                comp = client.chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": short_body}])
                answer = comp.choices[0].message.content
            except Exception as e:
                answer = "⚠️ *החיבור ל-Grok נכשל.*\nהאם הגדרת מפתח API חוקי והזנת כרטיס אשראי באתר של xAI?"

        if answer:
            send_whatsapp(From, answer)
        return

# ==========================================
# MODULE 7: ENTRY POINT (FASTAPI)
# ==========================================
app = FastAPI()

@app.post("/webhook")
async def whatsapp_webhook(background_tasks: BackgroundTasks, From: str = Form(...), Body: str = Form(""),
                           NumMedia: str = Form("0"), MediaUrl0: str = Form(None), MediaContentType0: str = Form(None)):
    """Receives webhook requests from Twilio and passes them to the background worker."""
    background_tasks.add_task(background_worker, From, Body, NumMedia, MediaUrl0, MediaContentType0)
    return Response(content="<Response></Response>", media_type="text/xml")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)