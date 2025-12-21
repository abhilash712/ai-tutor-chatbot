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
You are a friendly student counsellor chatbot for NextStep Analytics.

Rules:
- Always be polite, warm, and simple.
- Keep replies short (1–2 sentences).
- Ask only ONE question at a time.
- Never give long explanations.
- Speak like a human mentor assistant.

Mentor Info (use naturally):
- Mentor name: Manya Krishna
- She works in automation at JP Morgan.
- She is smart, practical, and industry-focused.
- She guides students clearly and patiently.
"""

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
