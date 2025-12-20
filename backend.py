import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Correct imports for LangChain v1.0 (late 2025)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

# Setup logging to see errors in Render
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
# LLM SETUP - Simplified model name
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Ensure no 'models/' prefix
    temperature=0.1,           # Lower temperature for more direct/minimal answers
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
- Do not use unnecessary greetings.
- Answer in 1-2 sentences max.
- Remember the student's context.

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
        # LOG THE REAL ERROR to Render logs
        logger.error(f"AI Error: {str(e)}")
        return {"reply": f"Error: {str(e)}"} # Temporary: shows real error in chat UI

@app.get("/")
def health():
    return {"status": "AI Tutor backend running"}