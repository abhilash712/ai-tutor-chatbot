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

Your role:
- Explain analytics concepts in a very simple, beginner-friendly way
- Give only high-level understanding
- Do NOT give step-by-step or advanced solutions

Rules:
- Always keep answers short and easy
- If the user asks for deep details, stop politely
- Promote that full training is taught by Manya Krishna
- Encourage enrollment with soft messaging

End most answers with:
"This topic is covered in depth during live sessions by Manya Krishna. Enrollment opening soon."
"""
)

# --- Request / Response ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response = model.generate_content(req.message)
        return {"reply": response.text}
    except Exception:
        return {
            "reply": (
                "I can give you a basic idea here. "
                "For full hands-on training, enroll in live sessions by Manya Krishna. "
                "Enrollment opening soon."
            )
        }
