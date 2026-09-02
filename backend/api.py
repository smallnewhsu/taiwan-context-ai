from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.asr_service import ASRService, SUPPORTED_EXTENSIONS
from backend.context_engine import Candidate, ContextEngine
from backend.feedback_service import FeedbackService
from backend.rewrite_service import RewriteService
from backend.vision_service import SUPPORTED_IMAGE_EXTENSIONS, VisionService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = PROJECT_ROOT / "datasets" / "taiwan_context" / "glossary.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024
FRONTEND_PATH = PROJECT_ROOT / "frontend"

app = FastAPI(title="Taiwan Context Engine API", version="0.2.0")
engine = ContextEngine(glossary_path=GLOSSARY_PATH, prompt_tolerance=-0.05)
asr_service = ASRService()
feedback_service = FeedbackService()
rewrite_service = RewriteService()
vision_service = VisionService()

if FRONTEND_PATH.exists():
    app.mount("/app", StaticFiles(directory=FRONTEND_PATH, html=True), name="app")


class CandidateInput(BaseModel):
    text: str
    confidence: float


class ContextRequest(BaseModel):
    baseline: CandidateInput
    prompt: CandidateInput


class FeedbackRequest(BaseModel):
    audio_file: str = Field(min_length=1, max_length=255)
    baseline_text: str = Field(min_length=1, max_length=1000)
    prompt_text: str = Field(min_length=1, max_length=1000)
    selected_source: Literal["baseline", "prompt", "manual"]
    corrected_text: str = Field(min_length=1, max_length=1000)
    reasons: list[str] = Field(default_factory=list)
    consent_to_dataset: bool = False


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer_note: str = Field(default="", max_length=1000)


class RewriteRequest(BaseModel):
    original_text: str = Field(min_length=1, max_length=2000)
    audience: Literal["elder", "family", "friend", "formal"]
    tone: Literal["warm", "clear", "polite", "concise"]
    scenario: str = Field(default="", max_length=1000)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/app/")


def make_decision(baseline_data, prompt_data):
    baseline = Candidate(
        source="baseline",
        text=baseline_data["text"],
        confidence=baseline_data["confidence"],
    )
    prompt = Candidate(
        source="prompt",
        text=prompt_data["text"],
        confidence=prompt_data["confidence"],
    )
    return engine.decide(baseline, prompt)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Taiwan Context Engine",
        "version": "0.2.0",
    }


@app.get("/llm/health")
def llm_health():
    return rewrite_service.health()


@app.get("/vision/health")
def vision_health():
    return vision_service.health()


@app.post("/vision/analyze")
async def analyze_image(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支援的格式。允許格式: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="圖片內容為空")
    if len(content) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="圖片超過15 MB限制")

    try:
        return await run_in_threadpool(
            vision_service.analyze_bytes,
            content,
            suffix,
            file.filename or "image",
        )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"圖片處理失敗: {exc}") from exc


@app.post("/expression/rewrite")
async def rewrite_expression(request: RewriteRequest):
    try:
        return await run_in_threadpool(
            rewrite_service.rewrite,
            request.original_text,
            request.audience,
            request.tone,
            request.scenario,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/context/select")
def select_candidate(request: ContextRequest):
    decision = make_decision(
        request.baseline.dict(), request.prompt.dict()
    )
    return decision.to_dict()


@app.post("/feedback", status_code=201)
def save_feedback(request: FeedbackRequest):
    return feedback_service.save(request.dict())


@app.get("/feedback/pending")
def list_pending_feedback():
    items = feedback_service.list_pending()
    return {"count": len(items), "items": items}


@app.post("/feedback/{feedback_id}/review")
def review_feedback(feedback_id: str, request: ReviewRequest):
    try:
        return feedback_service.review(
            feedback_id, request.decision, request.reviewer_note
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/speech/interpret")
async def interpret_speech(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"不支援的格式。允許格式: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="音檔內容為空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="音檔超過25 MB限制")

    try:
        asr_result = await run_in_threadpool(
            asr_service.interpret_bytes, content, suffix
        )
        decision = make_decision(asr_result["baseline"], asr_result["prompt"])
        return {
            "filename": file.filename,
            "audio_duration_seconds": asr_result["audio_duration_seconds"],
            "dual_pass_seconds": asr_result["dual_pass_seconds"],
            "baseline": asr_result["baseline"],
            "prompt": asr_result["prompt"],
            "decision": decision.to_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"語音處理失敗: {exc}") from exc
