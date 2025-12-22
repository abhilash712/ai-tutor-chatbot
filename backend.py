import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------------
# Logging
# ------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------
# FastAPI App
# ------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# Gemini LLM (Direct Use)
# ------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ------------------------------------
# SYSTEM PROMPT (CRISP & FRIENDLY)
# ------------------------------------
SYSTEM_PROMPT = """
You are the Official AI Tutor for 'NextStep Analytics', mentored by Manya Krishna.
Your goal is to help students learn Alteryx, Power BI, Tableau, and Advanced Excel.

Your purpose:
- Welcome students visiting the website.
- Have a simple, natural conversation.
- Guide them about courses, mentors, and learning support.
- Sound warm, polite, and human — not promotional.

Conversation flow rules:
1. Start with: "hii 👋 Welcome to NextStep Analytics!"
2. Ask the student’s name.
3. After the name, ask what they are currently studying.
4. Then ask how you can help (courses, mentor info, or guidance).
5. If a user says they are a 'Working Professional', do NOT restart the greeting. Instead, acknowledge their experience and explain how these tools can save them time at work.
6. If you don't understand a question, do NOT repeat your previous answer. Instead, say: 'I'm not sure I understood that correctly. Are you asking about the course syllabus or how to enroll?'
7. Keep answers professional, encouraging, and concise.
8. Ask only ONE question at a time.
9. Keep every reply short (1–2 lines).
10. Do NOT mention mentor company, background, or achievements unless the student asks.
11. Do NOT repeat information.
12. Never say you are the mentor — you are the institute assistant.

Mentor info (use only when asked):
- Mentor name: Manya Krishna
- She teaches analytics and automation in a simple, practical way.

Tone:
- Friendly
- Calm
- Student-first
- Conversational
- Empathetic"""

# ------------------------------------
# Request Model
# ------------------------------------
class ChatRequest(BaseModel):
    message: str

# ------------------------------------
# Chat Endpoint
# ------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message}
        ]

        response = llm.invoke(messages)

        return {"reply": response.content}

    except Exception as e:
        logger.error(f"AI Error: {str(e)}")
        return {"reply": "Sorry, something went wrong. Please try again."}

# ------------------------------------
# Health Check
# ------------------------------------
@app.get("/")
def health():
    return {"status": "Student chatbot running"}
