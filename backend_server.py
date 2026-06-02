from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import the existing graph and LangChain classes from the user's pipeline
from empathyagentpipeline import graph, AgentState
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="MediRAG API")

# Configure CORS for the Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: Optional[int] = 1 # Bypassing login for testing, default to user 1

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    
    # Check if this is a new session
    state = graph.get_state(config)
    if not state.values:
        print(f"[Backend] Initializing new session: {req.session_id}")
        initial_state = {
            "user_id": req.user_id,
            "messages": [],
            "current_symptoms": {},
            "followup_count": 0,
            "session_active": True,
            "journal_prompt_text": "",
            "trend_message": "",
            "past_history_context": "No previous history found." # Or call get_past_sessions_context(req.user_id)
        }
        # Run until it hits the first interrupt
        for _ in graph.stream(initial_state, config):
            pass
            
    new_messages = []
    
    print(f"[Backend] Resuming graph for session {req.session_id} with input: {req.message}")
    # Resume the graph with the user's input
    try:
        for event in graph.stream(Command(resume=req.message), config, stream_mode="updates"):
            for node, values in event.items():
                if isinstance(values, dict) and "messages" in values:
                    # In LangGraph, the node output contains the new messages it generated
                    for msg in values["messages"]:
                        if isinstance(msg, AIMessage):
                            role = msg.name if msg.name else ("Doctor" if node in ["doctor", "critic", "journal"] else "Assistant")
                            new_messages.append({"role": role, "text": msg.content})
    except Exception as e:
        print(f"[Backend] Error executing graph: {e}")
        return {"messages": [{"role": "System", "text": f"Sorry, an error occurred processing your message. ({str(e)})"}], "session_ended": False}
    
    # Check if the graph is waiting again, or finished
    current_state = graph.get_state(config)
    session_ended = not current_state.next
    
    return {
        "messages": new_messages,
        "session_ended": session_ended
    }
