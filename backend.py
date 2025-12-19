from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import google.generativeai as genai

# --- App setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini setup ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-pro",
    system_instruction="""
You are a learning assistant for NextStep Analytics.

You are a friendly learning assistant for NextStep Analytics.

Your goal:
- Gently guide students who are exploring analytics courses
- Start with a warm greeting
- Ask the student’s name if not known
- Ask what subject they are interested in
- Give only a very basic, beginner-level explanation
- Create curiosity for full training

Rules:
- Never give step-by-step solutions
- Never give advanced or complete answers
- Keep responses short and friendly
- Sound human and conversational

Conversation behavior:
- If the user says "hi" or greets, ask their name
- If the user asks a topic directly, give a short intro (2–3 lines max)
- End explanations with a soft promotion:
  "This is taught in detail during live sessions by Manya Krishna. Enrollment opening soon."

Tone:
- Warm
- Encouraging
- Student-focused
""",
)

# --- Request / Response ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_msg = req.message.lower().strip()

    # 1️⃣ Greeting handling (hard rule)
    if user_msg in ["hi", "hello", "hey", "hii", "hai"]:
        return {
            "reply": (
                "Hi 👋 Welcome to NextStep Analytics!\n\n"
                "I’m here to help you explore analytics courses.\n"
                "May I know your name?"
            )
        }

    # 2️⃣ Name provided (simple heuristic)
    if len(user_msg.split()) <= 3 and user_msg.isalpha():
        return {
            "reply": (
                f"Nice to meet you, {req.message} 😊\n\n"
                "What would you like to learn?\n"
                "Alteryx, Power BI, Tableau, or Excel?"
            )
        }

    # 3️⃣ Otherwise → Gemini
    try:
        response = model.generate_content(req.message)
        return {"reply": response.text}
    except Exception:
        return {
            "reply": (
                "I can give you a basic idea here.\n"
                "For full hands-on training, enroll in live sessions by Manya Krishna.\n"
                "Enrollment opening soon."
            )
        }


