from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow browser to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    message: str

@app.post("/chat")
def chat(q: Question):
    user_message = q.message

    reply = f"I received your question: '{user_message}'. Tutor reply coming soon."

    return {"reply": reply}
