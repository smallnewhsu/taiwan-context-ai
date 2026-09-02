from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from backend.context_engine import Candidate, ContextEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "taiwan_context"
    / "glossary.json"
)

app = FastAPI(
    title="Taiwan Context Engine API",
    version="0.1.0",
)

engine = ContextEngine(
    glossary_path=GLOSSARY_PATH,
    prompt_tolerance=-0.05,
)


class CandidateInput(BaseModel):
    text: str
    confidence: float


class ContextRequest(BaseModel):
    baseline: CandidateInput
    prompt: CandidateInput


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Taiwan Context Engine",
        "version": "0.1.0",
    }


@app.post("/context/select")
def select_candidate(request: ContextRequest):
    baseline = Candidate(
        source="baseline",
        text=request.baseline.text,
        confidence=request.baseline.confidence,
    )

    prompt = Candidate(
        source="prompt",
        text=request.prompt.text,
        confidence=request.prompt.confidence,
    )

    decision = engine.decide(baseline, prompt)

    return decision.to_dict()