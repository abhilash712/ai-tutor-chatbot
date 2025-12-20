import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Imports for LangChain v1.0 classic modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

# Logging to help you see any future errors in Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LLM SETUP - Updated to 2025 Stable Model
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ Updated stable model name
    temperature=0.1,           # Low temp for direct, minimal answers
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --------------------------------------------------
# PROMPT - Updated for minimal, concise answers
# --------------------------------------------------
template = """
You are a minimal AI tutor from NextStep Analytics.

Rules:
- Give very short, direct, and minimal answers.
- 1-2 sentences maximum.
- Do not say "Hello" or "How can I help" in every message.
- Be helpful but extremely brief.

Chat history: {chat_history}
Student: {user_input}
Tutor:"""

prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain.run(request.message)
        return {"reply": response}
    except Exception as e:
        logger.error(f"AI Error: {str(e)}") # Shows the specific error in Render logs
        return {"reply": f"Model Error: {str(e)}"}

@app.get("/")
def health():
    return {"status": "AI Tutor backend running"}