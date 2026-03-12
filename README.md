# 🚀 AI Center: Enterprise Multi-Agent WhatsApp Architecture

**Developed by:** Roye Schechter

## 📌 Overview
AI Center is a highly scalable, multi-modal AI gateway accessible directly via WhatsApp. Unlike standard single-model chatbots, this system features a **Semantic Routing Engine** that autonomously directs user queries to the most efficient LLM based on context, complexity, and media presence. 

The infrastructure boasts a built-in **RAG (Retrieval-Augmented Generation)** pipeline for long-term memory, multi-modal processing (audio and image analysis), and seamless integration with the world's leading AI providers.

---
## 🏗 System Flow Architecture

```text
                         ┌──────────────────────────────┐
                         │        WhatsApp Users        │
                         │  Text / Audio / Image / PDF  │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │           Twilio API         │
                         │        WhatsApp Webhook      │
                         └───────────────┬──────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │         FastAPI Backend      │
                         │    State & Route Controller  │
                         └───────────────┬──────────────┘
                                         │
               ┌─────────────────────────┴─────────────────────────┐
               │ User selects Direct Agent (1-5) or Smart Agent (0)│
               ▼                                                   ▼
 ┌───────────────────────────┐                       ┌───────────────────────────┐
 │     DIRECT AGENTS (1-5)   │                       │    SMART MANAGER (0)      │
 │ Bypass Memory & Routing   │                       │ Autonomous Orchestration  │
 └──────┬───────────┬────────┘                       └─────────────┬─────────────┘
        │           │                                              │
        ▼           ▼                                              ▼
  ┌──────────┐ ┌──────────┐                        ┌──────────────────────────────┐
  │  OpenAI  │ │Anthropic │                        │     Knowledge Layer (RAG)    │
  │  GPT-4o  │ │ Claude   │                        │ PDF Ingestion & Pinecone DB  │
  └──────────┘ └──────────┘                        └──────────────┬───────────────┘
        │           │                                             │
        ▼           ▼                                             ▼
  ┌──────────┐ ┌──────────┐                        ┌──────────────────────────────┐
  │  Gemini  │ │  Meta    │                        │  Intent Classifier (Llama)   │
  │  Direct  │ │  Llama   │                        │ Fast text OR Complex media?  │
  └──────────┘ └──────────┘                        └──────────────┬───────────────┘
        │           │                                             │
        └─────┬─────┘                             ┌───────────────┴───────────────┐
              │                                   ▼                               ▼
              │                     ┌──────────────────────────┐    ┌──────────────────────────┐
              │                     │        Groq Llama        │    │      Google Gemini       │
              │                     │ (Fast Text Inference)    │    │ (Vision, RAG & Imagen 3) │
              │                     └─────────────┬────────────┘    └──────────────┬───────────┘
              │                                   │                                │
              └───────────────────────────────────┼────────────────────────────────┘
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │       Twilio Response        │
                                   │  Auto-Chunking (>1500 chars) │
                                   └──────────────────────────────┘
                                   
```                                   
## ✨ Key Features & Technical Highlights

### 1. 🧠 Smart Autonomous Routing (Zero-Agent)
The system utilizes a lightweight, ultra-fast LLM (`Llama-3.3-70b` via Groq) as an "Intent Classifier". 
* It analyzes incoming prompts in milliseconds.
* Standard text queries are handled locally by the fast model to save costs and reduce latency.
* Complex queries or messages containing media (Audio/Images/PDFs) are automatically dynamically routed to a heavy-duty multi-modal agent (`Gemini 2.0 Flash/1.5 Flash`).

### 2. 📚 RAG Pipeline & Long-Term Memory
Built a complete ingestion and retrieval pipeline for document chatting.
* **Ingestion:** Downloads PDFs sent via WhatsApp, extracts text using `PyPDF2`, chunks the data, generates vector embeddings via Google's embedding model, and stores them in **Pinecone**.
* **Retrieval:** Contextually retrieves top-k relevant chunks to augment the prompt before sending it to the LLM, giving the AI long-term memory of user documents.

### 3. 🎤 Multi-Modal Processing
Users can send voice notes and images directly in WhatsApp. The backend intercepts Twilio media URLs, downloads the binary files, and utilizes Gemini's native multi-modal capabilities to transcribe, summarize, and analyze the content on the fly.

### 4. ⚙️ Resilient Infrastructure & API Management
Designed with production-grade constraints in mind:
* **Twilio Payload Management:** Implemented an automated chunking algorithm to bypass Twilio's strict 1600-character limit, ensuring long LLM responses (like document summaries) are cleanly split and delivered sequentially without failing.
* **State Management:** Uses `SQLite` as a persistence layer to track the active agent state for each user phone number, allowing seamless context-switching between different AI models.
* **Fault Tolerance:** Comprehensive `try/except` blocks around external API calls prevent server crashes during quota limits (e.g., HTTP 429) or missing API keys, returning graceful error messages to the user.

### 5. 🌐 Multi-Provider Gateway
A unified interface allowing users to manually switch between top-tier AI models by simply sending a digit:
* `1` - **OpenAI** (GPT-4o)
* `2` - **Anthropic** (Claude 3.5 Sonnet)
* `3` - **Google** (Gemini)
* `4` - **Meta** (Llama 3.3 via Groq)
* `5` - **xAI** (Grok)
### 6. 🎨 Image Generation & Vision (NEW)
* **Text-to-Image:** Integrated with Google's Imagen 3. Simply type "צייר" (Draw) followed by your prompt to generate and receive high-quality images directly in WhatsApp.
* **Computer Vision:** Send images to the bot, and Agent 0 will analyze and describe them using Gemini 2.0 Flash.

---

## 🛠️ Tech Stack
* **Backend Framework:** Python, FastAPI, Uvicorn (Asynchronous processing)
* **Databases:** Pinecone (Vector DB), SQLite (Relational DB)
* **AI & LLMs:** Google GenAI SDK (Gemini & Imagen 3), Groq API, OpenAI, Anthropic, xAI
 * * **Integrations:** Twilio API (WhatsApp Webhooks)
* **Utilities:** PyPDF2, Python-dotenv, Requests

## 💬 Example Interaction

User (WhatsApp):
"Summarize the PDF I just sent."

System Flow:
1. Twilio webhook receives message
2. FastAPI downloads the PDF
3. RAG pipeline processes document
4. Relevant chunks retrieved from Pinecone
5. Gemini generates contextual summary

Bot Response:
"Here is a summary of your document..."
---
## 🔮 Future Improvements

• Web dashboard for monitoring AI routing  
• Docker containerization  
• Redis caching layer  
• Streaming responses  
• Observability (Langfuse / OpenTelemetry)

## 🚀 Getting Started

### Prerequisites
1. Python 3.9+
2. A Twilio Developer account with a WhatsApp Sandbox.
3. API Keys for Pinecone, Groq, and Gemini (OpenAI, Anthropic, and xAI are optional).

### 1. Installation

Clone the repository to your local machine:
```bash
git clone https://gitlab.com/roye.schechter-group/roye.schechter-project.git
cd roye.schechter-project
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```
2. Configuration

Create a .env file in the root directory and populate it with your API keys. This file is ignored by Git for security:
```bash
TWILIO_ACCOUNT_SID=your_twilio_sid_here
TWILIO_AUTH_TOKEN=your_twilio_token_here
GEMINI_API_KEY=your_google_gemini_key_here
GROQ_API_KEY=your_groq_api_key_here
PINECONE_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=nexus-knowledge
```
3. Execution

To run the server locally for testing (ensure you have Ngrok running for the webhook):

```bash
uvicorn main:app --reload
```
For production deployment (Render/Cloud), the start command should be:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```
Developed by Roye Schechter ⚡