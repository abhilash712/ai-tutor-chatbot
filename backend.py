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
    allow_origins=["*"],   # Netlify access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LLM SETUP (Gemini)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.4
)

# --------------------------------------------------
# MEMORY (THIS FIXES YOUR ISSUE)
# --------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --------------------------------------------------
# PROMPT (VERY IMPORTANT)
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
You are a friendly analytics tutor from NextStep Analytics.

Your role:
- Help beginners understand analytics tools
- Speak in simple, clear language
- Be warm and student-friendly
- Never sound robotic
- Never repeat the same sentence again and again

Teaching rules:
- If the user greets, greet back and ask their name
- If name is known, use the name
- Ask what subject they want (Alteryx, Power BI, Tableau, Excel)
- Give only a basic explanation (not deep training)
- After every answer, ask ONE short follow-up question

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
# API MODEL
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = chain.run(request.message)
        return {"reply": response}
    except Exception as e:
        return {
            "reply": "I’m having a small issue right now. Please try again in a moment 🙂"
        }

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "AI Tutor Backend Running"}



