from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import tempfile, uuid
from chatbot import ChatWithPDF
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],    
)
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/upload")
async def upload_file_and_bot_object(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    bot = ChatWithPDF(pdf_path=tmp_path)
    bot_id = str(uuid.uuid4())
    sessions[bot_id] = bot

    return {"message": "PDF uploaded successfully", "session_id": bot_id}

@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    bot = sessions.get(request.session_id)
    if not bot:
        return {"reply": "No active session. Upload PDF first."}

    response = ""
    for chunk in bot.ask_question(request.message, request.session_id):
        response += chunk

    return {"reply": response}

# safer mount
app.mount("/", StaticFiles(directory="static", html=True), name="static")