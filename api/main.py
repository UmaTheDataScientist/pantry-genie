from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pantry_genie.agent import build_agent, chat

app = FastAPI(
    title="PantryGenie API",
    description="AI-powered vegan recipe assistant",
    version="1.0.0"
)

# ── CORS (needed for Streamlit to talk to FastAPI) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Build agent once at startup ────────────────────────────
agent = build_agent()

# ── Request/Response schemas ───────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    thread_id: str

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "🧞 PantryGenie is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # Set thread-local storage so tools use correct user files
        from pantry_genie import tools as t
        import threading
        t._thread_local.thread_id = request.thread_id

        response = chat(
            user_input=request.message,
            agent=agent,
            thread_id=request.thread_id
        )
        return ChatResponse(
            response=response,
            thread_id=request.thread_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pantry")
def update_pantry_endpoint(ingredients: list[str]):
    """Directly update pantry contents."""
    from pantry_genie.tools import update_pantry
    result = update_pantry.invoke(", ".join(ingredients))
    return {"status": result}

@app.get("/pantry")
def get_pantry_endpoint():
    """Get current pantry contents."""
    from pantry_genie.tools import get_pantry_contents
    result = get_pantry_contents.invoke("")
    return {"pantry": result}