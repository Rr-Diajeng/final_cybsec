"""
inference_api/main.py
=====================
FastAPI inference server for the phishing detection pipeline.

Endpoints:
  POST /classify          — classify a single email
  POST /classify/batch    — classify a list of emails
  GET  /results           — return all stored classification results
  GET  /healthz           — health check

The pipeline for each email:
  1. Validate input
  2. GuardrailChecker (NeMo / regex)  →  BLOCKED → store & return immediately
  3. Ollama LLM inference              →  PHISHING | SAFE
  4. Persist result to SQLite
  5. Return structured JSON response

Run:
  uvicorn inference_api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Import guardrail checker (relative import safe when run via uvicorn) ──────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from guardrails.guardrail_checker import check_for_injection

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
OLLAMA_MODEL      = "phishing-detector"        # registered via `ollama create`
DB_PATH           = Path(__file__).parent / "results.db"
REQUEST_TIMEOUT   = 30.0  # seconds
MAX_BODY_WORDS    = 400
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert email security classifier. "
    "When given an email, respond with exactly one word: PHISHING or SAFE."
)

CLASSIFICATION_PROMPT = (
    "Classify the following email as PHISHING or SAFE. "
    "Reply with exactly one word: PHISHING or SAFE.\n\n{input_text}"
)


# ── Database helpers ───────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                id          TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                sender      TEXT,
                subject     TEXT,
                body_preview TEXT,
                label       TEXT NOT NULL,
                confidence  REAL,
                reasoning   TEXT,
                guardrail_status TEXT,
                latency_ms  REAL
            )
        """)
        conn.commit()
    log.info(f"Database initialised at {DB_PATH}")


def insert_result(row: dict) -> None:
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO classifications
            (id, timestamp, sender, subject, body_preview, label,
             confidence, reasoning, guardrail_status, latency_ms)
            VALUES (:id, :timestamp, :sender, :subject, :body_preview, :label,
                    :confidence, :reasoning, :guardrail_status, :latency_ms)
        """, row)
        conn.commit()


def fetch_results(limit: int = 200) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM classifications ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class EmailInput(BaseModel):
    sender:  Optional[str] = Field(default="", description="Sender email address")
    subject: Optional[str] = Field(default="", description="Email subject line")
    body:    str            = Field(...,        description="Email body text")


class ClassifyRequest(BaseModel):
    email: EmailInput


class BatchClassifyRequest(BaseModel):
    emails: list[EmailInput] = Field(..., max_length=50)


class ClassificationResult(BaseModel):
    id:               str
    timestamp:        str
    sender:           str
    subject:          str
    label:            str       # "PHISHING" | "SAFE" | "PROMPT_INJECTION"
    confidence:       float
    reasoning:        str
    guardrail_status: str
    latency_ms:       float


# ── LLM inference ──────────────────────────────────────────────────────────────

def truncate(text: str, max_words: int = MAX_BODY_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def build_input_text(email: EmailInput) -> str:
    body = truncate(email.body)
    if email.sender and email.subject:
        return (
            f"Sender: {email.sender} | "
            f"Subject: {email.subject} | "
            f"Body: {body}"
        )
    elif email.subject:
        return f"Subject: {email.subject} | Body: {body}"
    return f"Body: {body}"


async def call_ollama(input_text: str) -> tuple[str, str, float]:
    """
    Call the Ollama REST API.
    Returns (label, reasoning, confidence).
    label is always "PHISHING" or "SAFE".
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": CLASSIFICATION_PROMPT.format(input_text=input_text)},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

    content: str = data["message"]["content"].strip().upper()

    # Robustly parse label — the model should return exactly PHISHING or SAFE
    if re.search(r"\bPHISHING\b", content):
        label = "PHISHING"
        confidence = 0.95
    elif re.search(r"\bSAFE\b", content):
        label = "SAFE"
        confidence = 0.95
    else:
        # Unexpected response — flag for review
        log.warning(f"Unexpected Ollama response: {content!r}; defaulting to REVIEW")
        label = "REVIEW"
        confidence = 0.50

    return label, content, confidence


# ── Core classify logic ────────────────────────────────────────────────────────

async def classify_email(email: EmailInput) -> ClassificationResult:
    start = time.perf_counter()
    email_id  = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Guardrail check
    input_text      = build_input_text(email)
    guardrail_result = check_for_injection(input_text)

    if guardrail_result.status == "BLOCKED":
        latency = (time.perf_counter() - start) * 1000
        result_row = dict(
            id=email_id,
            timestamp=timestamp,
            sender=email.sender or "",
            subject=email.subject or "",
            body_preview=input_text[:200],
            label="PROMPT_INJECTION",
            confidence=1.0,
            reasoning=guardrail_result.reason,
            guardrail_status="BLOCKED",
            latency_ms=round(latency, 2),
        )
        insert_result(result_row)
        return ClassificationResult(**{k: v for k, v in result_row.items()
                                       if k != "body_preview"})

    # 2. LLM inference
    try:
        label, reasoning, confidence = await call_ollama(input_text)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}")

    latency = (time.perf_counter() - start) * 1000
    result_row = dict(
        id=email_id,
        timestamp=timestamp,
        sender=email.sender or "",
        subject=email.subject or "",
        body_preview=input_text[:200],
        label=label,
        confidence=confidence,
        reasoning=reasoning,
        guardrail_status=guardrail_result.status,
        latency_ms=round(latency, 2),
    )
    insert_result(result_row)
    return ClassificationResult(**{k: v for k, v in result_row.items()
                                   if k != "body_preview"})


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Phishing Email Detection API",
    description="LLM-powered email phishing classifier with NeMo Guardrails",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def health():
    """Check if the API and Ollama are reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "api": "ok",
        "ollama": "ok" if ollama_ok else "unreachable",
        "model": OLLAMA_MODEL,
    }


@app.post("/classify", response_model=ClassificationResult)
async def classify(req: ClassifyRequest):
    """Classify a single email."""
    return await classify_email(req.email)


@app.post("/classify/batch", response_model=list[ClassificationResult])
async def classify_batch(req: BatchClassifyRequest):
    """Classify up to 50 emails in one call."""
    import asyncio
    tasks = [classify_email(e) for e in req.emails]
    return await asyncio.gather(*tasks)


@app.get("/results", response_model=list[dict])
async def get_results(limit: int = 200):
    """Return the most recent classification results (used by the dashboard)."""
    return fetch_results(limit)
