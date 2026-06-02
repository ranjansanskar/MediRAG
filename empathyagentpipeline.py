# ============================================================
# CELL 1 — Imports and Environment Setup
# ============================================================
# %pip install langchain langchain-openai openai python-dotenv langchain-community
#             pypdf chromadb langchain-chroma langgraph langchain-huggingface
#             rank-bm25 sentence-transformers pydantic psycopg2-binary

import os
import re
import json
import uuid
import operator
import bcrypt
from contextlib import contextmanager
from time import perf_counter
from datetime import datetime
from typing import Optional, List, Dict, Any, Annotated

from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

import psycopg2                                          # ← Neon uses psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
os.environ["OPENROUTER_API_KEY"] = " "    # ← replace

# ── Neon DB connection string ─────────────────────────────────────────────────
# Get from: https://console.neon.tech → your project → Connection Details
# Format: postgresql://user:password@ep-xxxx.region.aws.neon.tech/dbname?sslmode=require
NEON_DSN = "postgresql://neondb_owner:npg_ugXV0cZTGpC8@ep-broad-leaf-aqz3ehl8.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require"


# ============================================================
# CELL 2 — Neon DB Setup
# ============================================================

_neon_pool = None

def get_neon_pool():
    global _neon_pool
    if _neon_pool is None:
        import time
        for attempt in range(3):
            try:
                _neon_pool = psycopg2.pool.SimpleConnectionPool(
                    1, 10,
                    NEON_DSN,
                    connect_timeout=15,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=5,
                    keepalives_count=5
                )
                break
            except psycopg2.OperationalError as e:
                if attempt < 2:
                    print(f"[DB] Pool creation failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    raise e
    return _neon_pool

@contextmanager
def neon_cursor(cursor_factory=None):
    pool_inst = get_neon_pool()
    conn = None
    
    for _ in range(3):
        conn = pool_inst.getconn()
        if conn.closed != 0:
            pool_inst.putconn(conn, close=True)
            continue
            
        try:
            with conn.cursor() as test_cur:
                test_cur.execute("SELECT 1")
            break
        except psycopg2.OperationalError:
            pool_inst.putconn(conn, close=True)
            continue
            
    if conn is None or conn.closed != 0:
        conn = pool_inst.getconn()

    try:
        if cursor_factory:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur
        else:
            with conn.cursor() as cur:
                yield cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        pool_inst.putconn(conn)

def init_neon_db():
    """Create sessions and users tables, and migrate schema if needed."""
    try:
        with neon_cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL PRIMARY KEY,
                    username      TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id        SERIAL PRIMARY KEY,
                    user_id   INTEGER,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    mood      TEXT,
                    anxiety   TEXT,
                    stress    TEXT,
                    sleep     TEXT,
                    energy    TEXT,
                    severity  TEXT
                )
            """)
            # Check current type of severity column
            cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name = 'sessions' AND column_name = 'severity'")
            res = cur.fetchone()
            if res and res[0] == 'text':
                cur.execute("""
                    ALTER TABLE sessions 
                    ALTER COLUMN severity TYPE INTEGER 
                    USING (NULLIF(severity, 'unknown')::integer)
                """)
        print("[DB] Neon tables ready.")
    except Exception as e:
        print(f"[DB] Initialization failed: {e}")

def register_user(username: str, password: str) -> Optional[int]:
    """Register a new user and return their user_id. Returns None if username exists."""
    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        with neon_cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash)
                VALUES (%s, %s) RETURNING id
            """, (username, pwd_hash))
            user_id = cur.fetchone()[0]
            return user_id
    except psycopg2.IntegrityError:
        return None
    except Exception as e:
        print(f"[DB] Registration failed: {e}")
        return None

def authenticate_user(username: str, password: str) -> Optional[int]:
    """Verify password and return user_id if valid, else None."""
    try:
        with neon_cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                user_id, stored_hash = row
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    return user_id
            return None
    except Exception as e:
        print(f"[DB] Authentication failed: {e}")
        return None

def get_past_sessions_context(user_id: int) -> str:
    """Retrieve the user's last 5 sessions to build context for the LLM."""
    try:
        with neon_cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT timestamp, mood, anxiety, stress, severity
                FROM sessions
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 5
            """, (user_id,))
            rows = cur.fetchall()
            
        if not rows:
            return "No previous history found."
            
        # Reverse to get chronological order for prompt
        rows = list(reversed(rows))
        context_str = "User's Past Sessions Context:\\n"
        for idx, row in enumerate(rows, 1):
            date_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            context_str += (f"Session {idx} ({date_str}): "
                            f"Mood: {row['mood']}, Anxiety: {row['anxiety']}, "
                            f"Stress: {row['stress']}, Severity: {row['severity']}\\n")
        return context_str
    except Exception as e:
        print(f"[DB] Fetching past sessions failed: {e}")
        return "Failed to fetch previous history."

def save_session(user_id: int, symptoms: dict):
    """Save a session's symptoms to Neon."""
    sev = symptoms.get("severity", "unknown")
    if sev == "unknown" or not str(sev).isdigit():
        sev = None
    else:
        sev = int(sev)

    try:
        with neon_cursor() as cur:
            cur.execute("""
                INSERT INTO sessions (user_id, mood, anxiety, stress, sleep, energy, severity)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id,
                symptoms.get("mood",    "unknown"),
                symptoms.get("anxiety", "unknown"),
                symptoms.get("stress",  "unknown"),
                symptoms.get("sleep",   "unknown"),
                symptoms.get("energy",  "unknown"),
                sev,
            ))
    except Exception as e:
        print(f"[DB] Save session failed: {e}")

def get_trend_message(user_id: int) -> str:
    """Compare last two sessions and return a warm trend message."""
    try:
        with neon_cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT severity FROM sessions
                WHERE user_id = %s AND severity IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 2
            """, (user_id,))
            rows = cur.fetchall()

        if len(rows) < 2:
            return ""

        last = rows[0]["severity"]
        prev = rows[1]["severity"]

        if last is not None and prev is not None:
            if last < prev:
                return "💚 You seem to be doing a little better than last time. That matters."
            elif last > prev:
                return "Today sounds harder than last time. That's okay — you showed up anyway."
        return ""
    except Exception as e:
        print(f"[DB] Trend fetch failed: {e}")
        return ""

# Run once on startup
init_neon_db()


# ============================================================
# CELL 3 — Load PDFs
# ============================================================

mentaldocs_path = "mentaldocs"
os.makedirs(mentaldocs_path, exist_ok=True)
all_docs = []

for file in os.listdir(mentaldocs_path):
    if file.endswith(".pdf"):
        path = os.path.join(mentaldocs_path, file)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": file, "page": i}
                ))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(all_docs)
print(f"[Docs] Loaded {len(chunks)} chunks from {len(all_docs)} pages.")


# ============================================================
# CELL 4 — Embeddings + BM25
# ============================================================

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

STOP_WORDS = {"the", "and", "is", "in", "to", "of", "it", "that", "on", "for", "with", "as", "was", "at", "by", "an", "be", "this", "which", "or", "but", "not", "are", "from", "they", "we", "an", "if", "you"}

def tokenize_for_sparse(text: str) -> list:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

chunk_texts      = [d.page_content for d in chunks]
tokenized_chunks = [tokenize_for_sparse(t) for t in chunk_texts]

bm25 = BM25Okapi(tokenized_chunks) if BM25Okapi and tokenized_chunks else None


# ============================================================
# CELL 5 — Chroma Vector DB
# ============================================================

if chunks:
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="doctor_kb",
        collection_metadata={"hnsw:space": "cosine"}
    )
    RETRIEVAL_K_DENSE = 6
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K_DENSE}
    )
else:
    db = None
    retriever = None
    print("[Warning] No chunks found — retrieval will return empty.")


# ============================================================
# CELL 6 — Hybrid Retrieval
# ============================================================

RETRIEVAL_K_SPARSE = 8
FUSION_TOP_K       = 8
FINAL_CONTEXT_TOP_K = 4
RRF_K              = 60
cross_encoder      = None

def get_reranker():
    global cross_encoder
    if CrossEncoder is None:
        return None
    if cross_encoder is None:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return cross_encoder

def retrieve_hybrid(query: str) -> List[Document]:
    if not retriever:
        return []

    # Dense
    dense_docs = retriever.invoke(query)
    dense_ranked = {doc.page_content: rank for rank, doc in enumerate(dense_docs)}

    # Sparse BM25
    sparse_ranked = {}
    if bm25:
        tokens  = tokenize_for_sparse(query)
        scores  = bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:RETRIEVAL_K_SPARSE]
        for rank, idx in enumerate(top_idx):
            sparse_ranked[chunk_texts[idx]] = rank

    # RRF fusion
    all_contents = set(dense_ranked) | set(sparse_ranked)
    fused = {}
    for content in all_contents:
        d = dense_ranked.get(content, len(dense_ranked) + 1)
        s = sparse_ranked.get(content, len(sparse_ranked) + 1)
        fused[content] = 1 / (RRF_K + d) + 1 / (RRF_K + s)

    top_contents = sorted(fused, key=fused.get, reverse=True)[:FUSION_TOP_K]
    doc_map      = {d.page_content: d for d in dense_docs}
    fused_docs   = [doc_map.get(c, Document(page_content=c)) for c in top_contents]

    # Optional re-ranking
    reranker = get_reranker()
    if reranker and fused_docs:
        pairs  = [(query, d.page_content) for d in fused_docs]
        scores = reranker.predict(pairs)
        fused_docs = [d for _, d in sorted(zip(scores, fused_docs), key=lambda x: x[0], reverse=True)]

    return fused_docs[:FINAL_CONTEXT_TOP_K]

def docs_to_context(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# CELL 7 — Multimodal Placeholder
# ============================================================

def process_multimodal_input(input_data: Any, input_type: str = "text") -> str:
    if input_type == "text":
        return input_data
    elif input_type == "audio":
        print("[System] Audio input not yet implemented.")
        return ""
    elif input_type == "image":
        print("[System] Image input not yet implemented.")
        return ""
    return str(input_data)


# ============================================================
# CELL 8 — LLMs, State, Pydantic Models
# ============================================================

llm_temperature_low = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.1,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
)

llm_temperature_med = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
)

class AgentState(TypedDict):
    user_id:            int
    messages:           Annotated[List[BaseMessage], operator.add]
    current_symptoms:   dict
    followup_count:     int
    session_active:     bool
    active_query:       str
    retrieved_docs:     list
    final_context:      str
    doctor_output:      str
    final_reply:        str
    journal_prompt_text: str                             # ← NEW
    trend_message:      str                              # ← NEW
    past_history_context: str                            # ← NEW for historical context

class SymptomExtraction(BaseModel):
    mood:      str = Field(description="User's mood. Use 'unknown' if not mentioned.")
    anxiety:   str = Field(description="Anxiety level. Use 'unknown' if not mentioned.")
    stress:    str = Field(description="Stress level. Use 'unknown' if not mentioned.")
    sleep:     str = Field(description="Sleep quality. Use 'unknown' if not mentioned.")
    energy:    str = Field(description="Energy levels. Use 'unknown' if not mentioned.")
    appetite:  str = Field(description="Appetite changes. Use 'unknown' if not mentioned.")
    duration:  str = Field(description="Duration of symptoms. Use 'unknown' if not mentioned.")
    severity:  str = Field(description="Severity from 1-10. Use 'unknown' if not mentioned.")

class PostCriticEval(BaseModel):
    is_grounded:    bool = Field(description="True if grounded and no medication mentioned.")
    feedback:       str  = Field(description="What needs to change if not grounded.")
    refined_output: str  = Field(description="Rewritten safe response if original failed.")


# ============================================================
# CELL 9 — Prompt Templates
# ============================================================

compounder_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a calm, empathetic mental health intake assistant.
Extract clinical symptoms into structured data. Mark missing fields as 'unknown'.
Example:
User: "I haven't slept for 2 days and I'm very sad. It's an 8/10."
Output: mood='sad', sleep='has not slept for 2 days', severity='8', rest 'unknown'.
"""),
    MessagesPlaceholder(variable_name="messages")
])
compounder_tool_llm = llm_temperature_low.with_structured_output(SymptomExtraction)

empathy_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, emotionally intelligent mental health companion.
The user has just shared something. Respond with genuine human warmth.

Here is some context about their previous sessions, if any:
{past_history_context}

Rules:
- Reflect back what they seem to be feeling (don't parrot their words)
- Validate without diagnosing ("That sounds really exhausting" not "You have anxiety")
- If the user explicitly asks about their previous sessions, you MAY explicitly acknowledge their past feelings/situations (e.g., "Last time you mentioned your breakup...") to show you remember.
- NEVER ask more than one question
- Keep it to 2-3 sentences max
- Do NOT mention symptom gathering or that a doctor will respond
- Match their emotional energy — if they're flat, don't be overly cheerful
"""),
    MessagesPlaceholder(variable_name="messages")
])

followup_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a calm, empathetic mental health assistant gathering context before a doctor consultation.

Decision rules:
1. If the core fields (mood/anxiety/stress), duration, and severity are NOT all gathered: 
   - Acknowledge what they shared and ask ONE gentle follow-up question about a missing field. Priority: mood → anxiety → stress → duration → severity.
2. If the core fields, duration, and severity ARE gathered, OR if the user refuses to answer more questions:
   - Check if you recently asked them a final confirmation (e.g., "Is there anything else you'd like to share, or should I consult the Doctor?").
   - If you HAVE NOT asked yet: Warmly and naturally ask if they want to share more, or if they are ready for some guidance from the Doctor. (e.g. "I feel like I have a good understanding of what you're going through. Are you ready for me to consult the Doctor for some guidance, or is there more on your mind?")
   - If you HAVE asked, and their reply indicates "no" or readiness to proceed: Respond EXACTLY with: NO_FOLLOWUP_NEEDED
   - If you HAVE asked, but they shared new information: Acknowledge the new information naturally in your own words, and then softly check in again to see if they're ready to proceed to the Doctor. DO NOT sound repetitive or robotic. Vary your phrasing (e.g., "Take your time. Should we pause here and see what the Doctor suggests, or do you want to keep talking?").

Never use the word 'symptom'. Keep it conversational.
Symptoms so far: {symptoms}
"""),
    MessagesPlaceholder(variable_name="messages")
])

doctor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a compassionate mental health support guide.
Provide educational guidance based ONLY on the provided medical reference.
Do NOT invent conditions. Do NOT mention or suggest any medication whatsoever.

USER'S PAST HISTORY:
{past_history_context}

SYMPTOMS: {symptoms}
MEDICAL KNOWLEDGE:
{context}

If the user is feeling well, happy, and not experiencing any distress or negative symptoms:
1. Warmly validate their positive state and celebrate their well-being.
2. Encourage them to keep up their healthy habits (like being around people, journaling, etc.).
3. End with a supportive sign-off (you do not need to include the iCall helpline if they are perfectly fine).

If the user is experiencing any distress, anxiety, low mood, or stress, structure your response exactly like this:
1. One sentence warmly validating how hard this must be.
2. Brief educational explanation of what these symptoms might indicate.
3. 2-3 specific actionable coping techniques from the context. Examples:
   - Anxiety → box breathing, 5-4-3-2-1 grounding
   - Low mood → behavioral activation, sunlight exposure
   - Sleep issues → sleep hygiene tips, body scan meditation
   - Stress → journaling, progressive muscle relaxation
4. End with exactly:
   "If these feelings stay heavy for more than a couple of weeks, speaking with a
   counsellor — even just once — can make a real difference. iCall (9152987821)
   offers free, confidential support in India."
"""),
    ("human", "{query}")
])

post_critic_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Post-Critic for a mental health chatbot.
Evaluate the doctor output against the retrieved context. Ensure it:
1. Does NOT mention or recommend any medication — not even supplements.
2. Is strictly grounded in the context (no hallucinated claims).
3. If the user is in distress, includes a warm specific referral (e.g. iCall) and at least one actionable coping technique. (If the user is completely happy and fine, this is not required).

If it fails any check, rewrite it safely.
CONTEXT: {context}
DOCTOR OUTPUT: {doctor_output}
""")
])
post_critic_tool_llm = llm_temperature_low.with_structured_output(PostCriticEval)

journal_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Based on the user's symptoms: {symptoms}

Give ONE short, warm journaling prompt (1-2 sentences) to help them reflect today.
Examples:
- "What's one small thing that brought you even a moment of comfort today?"
- "Write about a moment this week when you felt slightly more like yourself."
- "What's one thing you wish someone understood about how you're feeling right now?"

Keep it gentle and open-ended. Never ask them to rate or score themselves."""),
])


# ============================================================
# CELL 10 — Nodes
# ============================================================

def get_user_input_node(state: AgentState) -> dict:
    user_input = interrupt("Waiting for user input...")
    processed  = process_multimodal_input(user_input, "text")

    if processed.strip().lower() in ["exit", "quit", "bye"]:
        print("\nSession ended. Take care.")
        return {"session_active": False}

    return {"messages": [HumanMessage(content=processed)]}


CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "cut myself",
    "die", "abuse", "beaten", "hit me", "don't want to live"
]

def emergency_interceptor_node(state: AgentState) -> dict:
    """Check for high-risk phrases and intercept if necessary."""
    last_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_msg = m.content.lower()
            break
            
    for kw in CRISIS_KEYWORDS:
        if kw in last_msg:
            crisis_msg = "It sounds like you are going through an incredibly difficult time. Your life has value, and you don't have to face this alone. Please reach out to iCall at 9152987821 for free, confidential support. I am going to end this session now so you can focus on getting immediate help."
            return {
                "messages": [AIMessage(content=crisis_msg, name="System")],
                "session_active": False
            }
    
    return {}

def empathy_node(state: AgentState) -> dict:
    history_ctx = state.get("past_history_context", "No previous history found.")
    res = llm_temperature_med.invoke(
        empathy_prompt.format_messages(
            messages=state["messages"],
            past_history_context=history_ctx
        )
    )
    return {"messages": [AIMessage(content=res.content, name="Assistant")]}        # no direct print


def extract_symptoms_node(state: AgentState) -> dict:
    res: SymptomExtraction = compounder_tool_llm.invoke(
        compounder_prompt.format_prompt(messages=state["messages"])
    )
    new_symptoms     = {k: v for k, v in res.dict().items() if v and v != "unknown"}
    current_symptoms = state.get("current_symptoms", {})
    current_symptoms.update(new_symptoms)
    return {"current_symptoms": current_symptoms}


def followup_node(state: AgentState) -> dict:
    symptoms = state.get("current_symptoms", {})

    core_filled     = any(symptoms.get(f, "unknown") != "unknown" for f in ["mood", "anxiety", "stress"])
    duration_filled = symptoms.get("duration", "unknown") != "unknown"
    severity_filled = symptoms.get("severity", "unknown") != "unknown"

    if state.get("followup_count", 0) >= 15:
        transition_msg = "Thank you for sharing that with me. I've gathered enough context, and I'm going to consult with the Doctor now to provide you with some personalized guidance and coping techniques."
        return {
            "active_query": f"Mental health guidelines for {json.dumps(symptoms)}",
            "messages": [AIMessage(content=transition_msg, name="Assistant")]
        }

    unknown_fields = [
        f for f in ["mood", "anxiety", "stress", "sleep", "energy", "appetite", "duration", "severity"]
        if symptoms.get(f, "unknown") == "unknown"
    ]

    res = llm_temperature_med.invoke(followup_prompt.format_prompt(
        unknown_fields=", ".join(unknown_fields),
        symptoms=json.dumps(symptoms),
        messages=state["messages"]
    ))

    if "NO_FOLLOWUP_NEEDED" in res.content:
        transition_msg = "Thank you for sharing that with me. I've gathered enough context, and I'm going to consult with the Doctor now to provide you with some personalized guidance and coping techniques."
        return {
            "active_query": f"Mental health guidelines for {json.dumps(symptoms)}",
            "messages": [AIMessage(content=transition_msg, name="Assistant")]
        }

    return {
        "messages":       [AIMessage(content=res.content.strip(), name="Assistant")],
        "followup_count": state.get("followup_count", 0) + 1
    }


def retrieve_node(state: AgentState) -> dict:
    query = state.get("active_query", "")
    if not query:
        query = f"Mental health guidelines for {json.dumps(state.get('current_symptoms', {}))}"

    docs    = retrieve_hybrid(query)
    context = docs_to_context(docs)
    return {"retrieved_docs": docs, "final_context": context}


def doctor_node(state: AgentState) -> dict:
    history_ctx = state.get("past_history_context", "No previous history found.")
    res = llm_temperature_med.invoke(doctor_prompt.format_prompt(
        past_history_context=history_ctx,
        symptoms=json.dumps(state.get("current_symptoms", {})),
        context=state.get("final_context", ""),
        query="Provide a gentle educational summary based on these symptoms and context."
    ))
    return {"doctor_output": res.content}


def critic_node(state: AgentState) -> dict:
    eval_res: PostCriticEval = post_critic_tool_llm.invoke(post_critic_prompt.format_prompt(
        context=state.get("final_context", ""),
        doctor_output=state.get("doctor_output", "")
    ))

    final_reply = (
        eval_res.refined_output
        if not eval_res.is_grounded and eval_res.refined_output
        else state.get("doctor_output", "")
    )
    return {"final_reply": final_reply, "messages": [AIMessage(content=final_reply, name="Doctor")]}


def journal_node(state: AgentState) -> dict:
    symptoms = state.get("current_symptoms", {})

    # ── Save session to Neon ──────────────────────────────────────────────────
    save_session(state.get("user_id", 1), symptoms)

    # ── Generate journaling prompt ────────────────────────────────────────────
    res = llm_temperature_med.invoke(
        journal_prompt_template.format_messages(symptoms=json.dumps(symptoms))
    )
    prompt_text = f"\n💭 A reflection prompt for you:\n{res.content}"
    return {
        "journal_prompt_text": res.content,
        "messages": [AIMessage(content=prompt_text, name="Doctor")]
    }


# ============================================================
# CELL 11 — Routing
# ============================================================

def route_after_input(state: AgentState) -> str:
    if not state.get("session_active", True):
        return "end"
    return "emergency"

def route_after_emergency(state: AgentState) -> str:
    if not state.get("session_active", True):
        return "end"
    return "empathy"

def route_after_followup(state: AgentState) -> str:
    if state.get("active_query"):
        return "retrieve"
    return "get_input"


# ============================================================
# CELL 12 — Graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("get_input", get_user_input_node)
builder.add_node("emergency", emergency_interceptor_node)
builder.add_node("empathy",   empathy_node)
builder.add_node("extract",   extract_symptoms_node)
builder.add_node("followup",  followup_node)
builder.add_node("retrieve",  retrieve_node)
builder.add_node("doctor",    doctor_node)
builder.add_node("critic",    critic_node)
builder.add_node("journal",   journal_node)             # ← NEW

builder.add_edge(START, "get_input")
builder.add_conditional_edges(
    "get_input", route_after_input,
    {"emergency": "emergency", "end": END}
)
builder.add_conditional_edges(
    "emergency", route_after_emergency,
    {"empathy": "empathy", "end": END}
)
builder.add_edge("empathy",  "extract")
builder.add_edge("extract",  "followup")
builder.add_conditional_edges(
    "followup", route_after_followup,
    {"retrieve": "retrieve", "get_input": "get_input"}
)
builder.add_edge("retrieve", "doctor")
builder.add_edge("doctor",   "critic")
builder.add_edge("critic",   "journal")                 # ← changed from END
builder.add_edge("journal",  END)                       # ← NEW

graph = builder.compile(checkpointer=MemorySaver())


# ============================================================
# CELL 13 — Chat Runner
# ============================================================

def run_chat():
    print("Welcome to the Mental Health Assistant.")
    
    user_id = None
    while not user_id:
        print("\\n1. Log In")
        print("2. Sign Up")
        print("3. Exit")
        choice = input("Select an option (1/2/3): ").strip()
        
        if choice == "3":
            print("Take care.")
            return
            
        elif choice == "1":
            username = input("Username: ").strip()
            import getpass
            password = getpass.getpass("Password: ")
            
            uid = authenticate_user(username, password)
            if uid:
                print(f"\\nWelcome back, {username}!")
                user_id = uid
            else:
                print("\\nInvalid username or password. Please try again.")
                
        elif choice == "2":
            username = input("Choose a username: ").strip()
            import getpass
            password = getpass.getpass("Choose a password: ")
            
            if not username or not password:
                print("\\nUsername and password cannot be empty.")
                continue
                
            uid = register_user(username, password)
            if uid:
                print(f"\\nAccount created successfully! Welcome, {username}!")
                user_id = uid
            else:
                print("\\nUsername already exists. Please choose a different one or log in.")
        else:
            print("\\nInvalid choice. Please select 1, 2, or 3.")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    past_history_context = get_past_sessions_context(user_id)
    if past_history_context != "No previous history found.":
        print(f"\n[System] Found past history. The AI will remember your past sessions.")
        # If you want to see exactly what is sent to the LLM, uncomment the line below:
        # print(past_history_context)
    
    initial_state = {
        "user_id":            user_id,
        "messages":           [],
        "current_symptoms":   {},
        "followup_count":     0,
        "session_active":     True,
        "journal_prompt_text": "",
        "trend_message":      "",
        "past_history_context": past_history_context,
    }

    print("\\nMental Health Assistant started. Type 'exit' to quit.\\n")

    # ── Show trend from previous session ────────────────────────────────────
    trend = get_trend_message(user_id)
    if trend:
        print(f"{trend}\\n")

    for _ in graph.stream(initial_state, config):
        pass

    printed_count = 0
    while True:
        state    = graph.get_state(config)
        messages = state.values.get("messages", [])

        for msg in messages[printed_count:]:
            if isinstance(msg, AIMessage):
                prefix = getattr(msg, "name", None)
                if not prefix:
                    prefix = "Assistant" if state.next else "Doctor"
                print(f"\n{prefix}: {msg.content}\n")

        printed_count = len(messages)

        if not state.next:
            break

        user_input = input("You: ")
        for event in graph.stream(Command(resume=user_input), config):
            for node, values in event.items():
                if node == "empathy":
                    print("[System] Empathy response generated.")
                elif node == "extract":
                    symptoms = values.get("current_symptoms", {})
                    full = {k: symptoms.get(k, "unknown")
                            for k in ["mood","anxiety","stress","sleep","energy","appetite","duration","severity"]}
                    print(f"\n[System] Symptom state:\n{json.dumps(full, indent=2)}\n")
                elif node == "retrieve":
                    print("[System] Retrieving knowledge...")
                elif node == "doctor":
                    print("[System] Doctor is evaluating...")
                elif node == "critic":
                    print("[System] Reviewing safety...")
                elif node == "journal":
                    print("[System] Saving session to Neon & generating reflection prompt...")

if __name__ == "__main__":
    run_chat()
