import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate

# Logging setup
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
# LLM SETUP - Stable 2025 Model
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# --------------------------------------------------
# ENHANCED PROMPT - Education & Expertise
# --------------------------------------------------
template = """
You are the official AI Tutor for NextStep Analytics, representing Lead Mentor Manya Krishna.

### 🎓 MENTOR PROFILE & EDUCATION:
- **Academic Background:** Katta Manya Krishna holds an MBA in Finance from Osmania University (Wesley PG College)[cite: 46, 49, 50, 53].
- **Professional Path:** She transitioned from an Operations Tax Analyst at Franklin Templeton to a Data Analyst Team Lead at JP Morgan Chase.
- **Core Mastery:** She is an Alteryx Designer Core & Cloud Core Certified expert who has saved over 10,000 manual work hours through automation[cite: 70, 85, 86].
- **Technical Skills:** Expert in Alteryx, Tableau, UiPath RPA, and SQL-Basics[cite: 54].

### 🤖 YOUR COMMUNICATION RULES:
1. **The Greeting:** ALWAYS start your first response with "hii".
2. **Academic Authority:** When asked about studies, highlight her MBA from Osmania University to show her strong foundation in Finance and Analytics.
3. **Conciseness:** Provide enough detail about her studies and experience but keep the total reply to 2-3 sentences max.
4. **Call to Action:** Remind students that Manya teaches these corporate-level automation skills in her courses.

Conversation so far: {chat_history}
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
        logger.error(f"AI Error: {str(e)}")
        return {"reply": f"Error: {str(e)}"}

@app.get("/")
def health():
    return {"status": "AI Tutor backend running"}