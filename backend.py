import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LLM SETUP
# --------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # ✅ IMPORTANT
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# --------------------------------------------------
# MEMORY (CRITICAL)
# --------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
You are a friendly analytics tutor from NextStep Analytics.

Rules:
- Talk like a real human tutor
- Guide beginners step by step
- Remember user's name and subject choice
- Never repeat the same line
- Keep answers short and clear

Conversation so far:
{chat_history}

Student message:
{user_input}

Tutor reply:
"""
)

# --------------------------------------------------
# CHAIN
# --------------------------------------------------
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain.run(request.message)
        return {"reply": response}
    except Exception:
        return {
            "reply": "I’m having a small issue right now. Please try again in a moment 🙂"
        }

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "AI Tutor backend running"}



