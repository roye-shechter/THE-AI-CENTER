# 🌐 Universal AI WhatsApp Gateway
### Multi-Model AI Orchestration Platform for WhatsApp

**Author:** Roye Schechter  

A high-performance backend system that transforms WhatsApp into a **multi-model AI command center**, enabling users to interact with several leading LLMs through a single unified interface.

The platform provides **dynamic model routing, persistent conversational memory, and multimodal capabilities** — all built on a scalable asynchronous architecture.

---

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Framework-green.svg)
![WhatsApp](https://img.shields.io/badge/WhatsApp-Twilio%20Integration-darkgreen.svg)
![AI Models](https://img.shields.io/badge/AI-Gemini%20%7C%20Llama%20%7C%20GPT%20%7C%20Claude%20%7C%20Grok-orange.svg)

---

## 🚀 Project Overview

The **Universal AI Gateway** acts as an orchestration layer between users and multiple AI providers. Instead of integrating with each model separately, the system provides:

- A **single conversational interface** via WhatsApp
- Dynamic **on-the-fly model switching**
- **Persistent context memory** (maintains context even when switching models)
- **Multimodal processing** (Text, Voice, and Image generation/analysis)

---

## 🧠 Supported AI Models

| Provider | Model | Modality |
|--------|--------|--------|
| **Google** | Gemini 2.0 Flash | Text, Vision, Audio, Image Gen (Imagen 4) |
| **Meta** | Llama 3.3 70B (via Groq) | High-speed Text |
| **OpenAI** | GPT-4o | Text, Vision |
| **Anthropic** | Claude 3.5 Sonnet | Text, Vision |
| **xAI** | Grok-2 | Text |

*Note: The system architecture allows adding new models with minimal configuration.*

---

## 🏗️ System Architecture

```text
                           📱 User (WhatsApp)
                                  │
                                  ▼
                            ☁️ Twilio Webhook
                                  │
                                  ▼
                        ⚙️ FastAPI Gateway (main.py)
                                  │
      ┌─────────────┬─────────────┼─────────────┬─────────────┐
      ▼             ▼             ▼             ▼             ▼
   OpenAI       Anthropic        xAI          Google        Meta/Groq
  (GPT-4o)    (Claude 3.5)    (Grok-2)     (Gemini 2.0)   (Llama 3.3)
      │             │             │             │             │
      └─────────────┴─────────────┼─────────────┴─────────────┘
                                  ▼
                         🗄️ SQLite Database
                    (Session Management & Chat History)
```

The backend functions as a **model orchestration layer**, routing requests to the selected AI engine while seamlessly saving the interaction history to the database.

---

## ✨ Core Features

### 1. Multi-Model AI Switching
Users can dynamically switch between 5 different AI engines during the conversation using a clean, unified menu.

### 2. Persistent Conversation Memory
User context is stored using **SQLite**, mapped by their phone number. The history follows the user, allowing conversations to continue logically.

### 3. Multimodal Processing
The gateway supports advanced interactions:
- **Visual Intelligence:** Analyzing incoming images.
- **Generative Art:** Generating images dynamically.
- **Voice Recognition:** Processing WhatsApp audio messages.

### 4. Automatic Garbage Collection
A background asynchronous task runs continuously to clean up database sessions older than 24 hours, preventing database bloat and ensuring optimal performance.

---

## ⚙️ Tech Stack & Dependencies

- **Backend:** Python 3.11, FastAPI (async framework), Uvicorn
- **AI SDKs:** `google-genai`, `groq`, `openai`, `anthropic`
- **Infrastructure:** SQLite3, Twilio API, ngrok (for local tunneling)
- **Environment Management:** `python-dotenv`

---

## 📋 Prerequisites

Before running the project, ensure you have the following:
1. A **Twilio** account with a WhatsApp Sandbox activated.
2. Active API Keys from the respective AI providers (Google Studio, Groq, OpenAI, Anthropic, xAI).
3. Python 3.11+ installed on your local machine.

---

## 🔐 Security & Configuration

Sensitive credentials are strictly managed using environment variables. Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_google_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
XAI_API_KEY=your_xai_key

TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
```
*Never commit your `.env` file to version control.*

---

## 🧪 Running the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Expose the Localhost (Development):**
   ```bash
   ngrok http 8000
   ```
   *Update your Twilio Sandbox Webhook with the generated ngrok URL appended with `/webhook`.*

---

## 🔮 Future Improvements

- Implementing automatic **smart routing** (evaluating prompt complexity and routing to the cheapest/fastest model).
- Integrating a Vector Database (e.g., Pinecone/Chroma) for **long-term semantic memory** (RAG).
- Full deployment pipeline to cloud platforms like Render or AWS.

---

# 👨‍💻 Author

**Roye Schechter** Software developer focusing on **AI systems, backend infrastructure, and intelligent automation**.

---
*Engineering the future of AI communication.*