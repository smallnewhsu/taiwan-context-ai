from pathlib import Path
import json
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.asr_service import ASRService, SUPPORTED_EXTENSIONS
from backend.context_engine import Candidate, ContextEngine
from backend.feedback_service import FeedbackService
from backend.rewrite_service import RewriteService
from backend.vision_service import SUPPORTED_IMAGE_EXTENSIONS, VisionService
from backend.auth_service import AuthService
from backend.taigi_service import TaigiService
from backend.multimodal_service import MultimodalService
from backend.correction_memory import CorrectionMemory
from backend.speech_context_service import SpeechContextService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY_PATH = PROJECT_ROOT / "datasets" / "taiwan_context" / "glossary.json"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 15 * 1024 * 1024
FRONTEND_PATH = PROJECT_ROOT / "frontend"
AUTH_DATA_PATH = PROJECT_ROOT / "datasets" / "taiwan_context" / "auth"

app = FastAPI(title="Taiwan Context Engine API", version="0.2.0")
engine = ContextEngine(glossary_path=GLOSSARY_PATH, prompt_tolerance=-0.05)
asr_service = ASRService()
feedback_service = FeedbackService()
rewrite_service = RewriteService()
vision_service = VisionService()
auth_service = AuthService(AUTH_DATA_PATH)
taigi_service = TaigiService()
multimodal_service = MultimodalService()
correction_memory = CorrectionMemory()
speech_context_service = SpeechContextService()


@app.middleware("http")
async def protect_pages(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    public_paths = {"/", "/health", "/auth/status", "/auth/setup", "/auth/login", "/auth/logout"}
    if path in public_paths or path.startswith("/app/login"):
        return await call_next(request)
    user = auth_service.read_token(request.cookies.get(AuthService.COOKIE_NAME))
    if path.startswith("/app") and user is None:
        return RedirectResponse(url="/app/login/?next=" + request.url.path, status_code=303)
    if (path.startswith("/app/review") or path.startswith("/app/admin") or path in {"/docs", "/redoc", "/openapi.json"}) and (not user or user["role"] != "admin"):
        if path.startswith("/app"):
            return RedirectResponse(url="/app/?forbidden=1", status_code=303)
        return Response(status_code=403, content="Administrator access required")
    request.state.user = user
    return await call_next(request)

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
    audio_sha256: str = Field(default="", max_length=64)
    baseline_text: str = Field(min_length=1, max_length=1000)
    prompt_text: str = Field(min_length=1, max_length=1000)
    selected_source: Literal["baseline", "prompt", "manual"]
    corrected_text: str = Field(min_length=1, max_length=1000)
    reasons: list[str] = Field(default_factory=list)
    consent_to_dataset: bool = False
    feedback_type: str = Field(default="asr_transcription", max_length=40)
    ai_context: dict = Field(default_factory=dict)
    human_context: dict = Field(default_factory=dict)


class SpeechContextFeedbackRequest(BaseModel):
    audio_file: str = Field(min_length=1, max_length=255)
    audio_sha256: str = Field(min_length=64, max_length=64)
    transcription: str = Field(min_length=1, max_length=1000)
    ai_context: dict
    human_context: dict
    consent_to_dataset: bool = False


class ReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer_note: str = Field(default="", max_length=1000)


class RewriteRequest(BaseModel):
    original_text: str = Field(min_length=1, max_length=2000)
    audience: Literal["elder", "family", "friend", "formal"]
    tone: Literal["warm", "clear", "polite", "concise"]
    scenario: str = Field(default="", max_length=1000)


class TaigiRequest(BaseModel):
    original_text: str = Field(min_length=1, max_length=2000)
    audience: Literal["elder", "family", "friend", "formal"]
    scenario: str = Field(default="", max_length=1000)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=200)


class CreateUserRequest(LoginRequest):
    display_name: str = Field(min_length=1, max_length=80)
    role: Literal["user", "admin"] = "user"


class VisionTextFollowUpRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    image_context: dict
    relationship: str = Field(default="未提供", max_length=100)


class SpeechContextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    speaker_hint: str = Field(default="不確定", max_length=40)
    listener_hint: str = Field(default="不確定", max_length=40)
    extra_context: str = Field(default="", max_length=1000)


def current_user(request: Request):
    user = auth_service.read_token(request.cookies.get(AuthService.COOKIE_NAME))
    if not user:
        raise HTTPException(status_code=401, detail="請先登入")
    return user


def current_admin(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理者權限")
    return user


@app.get("/auth/status")
def auth_status(request: Request):
    return {"needs_setup": auth_service.needs_setup(), "user": auth_service.read_token(request.cookies.get(AuthService.COOKIE_NAME))}


@app.post("/auth/setup", status_code=201)
def setup_admin(request: CreateUserRequest, response: Response):
    if not auth_service.needs_setup():
        raise HTTPException(status_code=409, detail="系統已完成初始化")
    try:
        user = auth_service.create_user(request.username, request.password, request.display_name, "admin")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.set_cookie(AuthService.COOKIE_NAME, auth_service.issue_token(user), max_age=AuthService.SESSION_SECONDS, httponly=True, samesite="strict", secure=False, path="/")
    return {"user": user}


@app.post("/auth/login")
def login(request: LoginRequest, response: Response):
    user = auth_service.authenticate(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    response.set_cookie(AuthService.COOKIE_NAME, auth_service.issue_token(user), max_age=AuthService.SESSION_SECONDS, httponly=True, samesite="strict", secure=False, path="/")
    return {"user": user}


@app.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(AuthService.COOKIE_NAME, path="/")
    return {"status": "ok"}


@app.get("/auth/me")
def me(user=Depends(current_user)):
    return {"user": user}


@app.get("/admin/users")
def list_users(_=Depends(current_admin)):
    return {"items": auth_service.list_users()}


@app.post("/admin/users", status_code=201)
def create_user(request: CreateUserRequest, _=Depends(current_admin)):
    try:
        return {"user": auth_service.create_user(request.username, request.password, request.display_name, request.role)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/admin/dataset")
def dataset_overview(_=Depends(current_admin)):
    feedback_root = PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback"
    candidate_records = feedback_service._read_jsonl(feedback_root / "approved_feedback.jsonl")
    candidates = []
    for row in reversed(candidate_records):
        human_context = row.get("human_context") or {}
        candidates.append({
            "feedback_id": row.get("feedback_id"),
            "feedback_type": row.get("feedback_type", "transcription"),
            "audio_file": row.get("audio_file"),
            "reference_text": row.get("corrected_text") or row.get("reference_text"),
            "context_summary": human_context.get("possible_intent") or human_context.get("literal_meaning"),
            "reviewed_at": row.get("reviewed_at"),
            "raw_audio_included": bool(row.get("raw_audio_stored", False)),
        })
    release_root = PROJECT_ROOT / "datasets" / "taiwan_context" / "releases"
    released_records, releases = [], []
    if release_root.exists():
        for dataset_file in sorted(release_root.glob("*/dataset.jsonl")):
            count = 0
            for line in dataset_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("consent_verified") and row.get("human_reviewed"):
                    count += 1
                    released_records.append({"sample_id": row.get("sample_id"), "domain": row.get("domain"), "reference_text": row.get("reference_text"), "raw_audio_included": bool(row.get("raw_audio_included"))})
            releases.append({"version": dataset_file.parent.name, "approved_count": count})
    return {
        "approved_count": len(candidates),
        "candidate_count": len(candidates),
        "released_count": len(released_records),
        "release_count": len(releases),
        "releases": releases,
        "candidate_items": candidates[:200],
        "items": released_records[-100:],
    }


@app.get("/admin/services")
def service_overview(_=Depends(current_admin)):
    return {"items": [
        {"name": "Taiwan Context Engine", "status": "ok", "detail": "語境選擇與安全判斷可用"},
        {"name": "文字模型", **rewrite_service.health()},
        {"name": "視覺模型", **vision_service.health()},
        {"name": "語音辨識", "status": "ok", "detail": "Taiwan Tongues ASR CE 已設定；首次請求才載入模型"},
    ]}


def _read_summary(*relative_candidates):
    for relative in relative_candidates:
        path = PROJECT_ROOT / relative
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
    return {}


@app.get("/admin/overview")
def admin_overview(_=Depends(current_admin)):
    pending = feedback_service.list_pending()
    review_events = feedback_service._read_jsonl(
        PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback" / "review_events.jsonl"
    )
    approved = feedback_service._read_jsonl(
        PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback" / "approved_feedback.jsonl"
    )
    asr = _read_summary(
        "evaluation/asr_baseline/baseline_results_gpu_10.json",
        "evaluation/asr_baseline/prompt_results_gpu_10.json",
    )
    rewrite = _read_summary(
        "evaluation/rewrite/results/v0.2.3_gpu/summary.json",
        "evaluation/rewrite/results/v0.1_gpu/summary.json",
    )
    vision = _read_summary(
        "evaluation/vision/results/v0.1.1_gpu/summary.json",
        "evaluation/vision/results/v0.1_gpu/summary.json",
    )
    return {
        "feedback": {
            "pending_count": len(pending),
            "approved_review_count": sum(event.get("decision") == "approve" for event in review_events),
            "rejected_review_count": sum(event.get("decision") == "reject" for event in review_events),
            "dataset_approved_count": len(approved),
            "consented_pending_count": sum(bool(item.get("consent_to_dataset")) for item in pending),
        },
        "evaluations": {"speech": asr, "rewrite": rewrite, "vision": vision},
        "privacy": {
            "raw_audio_default_stored": False,
            "image_default_persisted": False,
            "dataset_requires_consent": True,
            "dataset_requires_human_review": True,
        },
    }


@app.get("/admin/feedback-history")
def feedback_history(_=Depends(current_admin)):
    submissions = feedback_service._read_jsonl(
        PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback" / "pending_feedback.jsonl"
    )
    events = feedback_service._read_jsonl(
        PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback" / "review_events.jsonl"
    )
    by_id = {item.get("feedback_id"): item for item in submissions}
    items = [{**by_id.get(event.get("feedback_id"), {}), **event} for event in reversed(events)]
    return {"count": len(items), "items": items[:200]}


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


@app.post("/vision/follow-up")
async def vision_follow_up(
    audio: UploadFile = File(...),
    image_context: str = Form(...),
    relationship: str = Form(default="未提供", max_length=100),
):
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in set(SUPPORTED_EXTENSIONS) | {".webm", ".ogg"}:
        raise HTTPException(status_code=415, detail=f"不支援的音檔格式: {suffix}")
    content = await audio.read()
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="音檔為空或超過 25 MB")
    try:
        context_data = json.loads(image_context)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="影像語境格式錯誤") from exc
    try:
        asr_result = await run_in_threadpool(asr_service.interpret_bytes, content, suffix)
        decision = make_decision(asr_result["baseline"], asr_result["prompt"])
        decision_data = decision.to_dict()
        answer = await run_in_threadpool(multimodal_service.answer, decision_data["selected_text"], context_data, relationship)
        return {"question": decision_data["selected_text"], "question_needs_confirmation": decision_data["needs_confirmation"], "asr": asr_result, "answer": answer, "flow": ["VLM", "Taiwan Tongues ASR CE", "Taiwan Context Engine", "LLM"]}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"多模態問答失敗: {exc}") from exc


@app.post("/vision/follow-up/text")
async def vision_text_follow_up(request: VisionTextFollowUpRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="問題內容不可為空白")
    try:
        answer = await run_in_threadpool(
            multimodal_service.answer,
            question,
            request.image_context,
            request.relationship,
        )
        return {
            "input_mode": "text",
            "question": question,
            "question_needs_confirmation": False,
            "answer": answer,
            "flow": ["VLM", "文字問題", "Taiwan Context Engine", "LLM"],
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文字追問處理失敗: {exc}") from exc


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


@app.post("/expression/taigi")
async def convert_to_taigi(request: TaigiRequest):
    try:
        return await run_in_threadpool(taigi_service.convert, request.original_text, request.audience, request.scenario)
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
    try:
        return feedback_service.save(request.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/speech/context-feedback", status_code=201)
def save_speech_context_feedback(request: SpeechContextFeedbackRequest):
    try:
        return feedback_service.save_context(request.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/feedback/pending")
def list_pending_feedback(_=Depends(current_admin)):
    items = feedback_service.list_pending()
    return {"count": len(items), "items": items}


@app.post("/feedback/{feedback_id}/review")
def review_feedback(feedback_id: str, request: ReviewRequest, _=Depends(current_admin)):
    try:
        submission = feedback_service.get_submission(feedback_id)
        event = feedback_service.review(
            feedback_id, request.decision, request.reviewer_note
        )
        if request.decision == "approve":
            correction = correction_memory.approve(submission, event)
            event["correction_memory_added"] = correction is not None
        return event
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

    import hashlib
    audio_sha256 = hashlib.sha256(content).hexdigest()
    approved_correction = correction_memory.find(audio_sha256)

    if approved_correction:
        corrected_text = approved_correction["corrected_text"]
        return {
            "filename": file.filename,
            "audio_sha256": audio_sha256,
            "audio_duration_seconds": None,
            "dual_pass_seconds": 0.0,
            "baseline": {"text": corrected_text, "confidence": 1.0, "inference_seconds": 0.0},
            "prompt": {"text": corrected_text, "confidence": 1.0, "inference_seconds": 0.0},
            "decision": {
                "selected_source": "human_review",
                "selected_text": corrected_text,
                "decision_confidence": 1.0,
                "needs_confirmation": False,
                "reasons": ["human_verified_correction"],
                "baseline_score": 1.0,
                "prompt_score": 1.0,
            },
            "correction_applied": True,
            "correction_source": "human_review",
            "context_override": approved_correction.get("human_context"),
        }

    try:
        asr_result = await run_in_threadpool(
            asr_service.interpret_bytes, content, suffix
        )
        decision = make_decision(asr_result["baseline"], asr_result["prompt"])
        return {
            "filename": file.filename,
            "audio_sha256": audio_sha256,
            "audio_duration_seconds": asr_result["audio_duration_seconds"],
            "dual_pass_seconds": asr_result["dual_pass_seconds"],
            "baseline": asr_result["baseline"],
            "prompt": asr_result["prompt"],
            "decision": decision.to_dict(),
            "correction_applied": False,
            "correction_source": None,
        }
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"語音處理失敗: {exc}") from exc


@app.post("/speech/context")
async def analyze_speech_context(request: SpeechContextRequest):
    try:
        return await run_in_threadpool(
            speech_context_service.analyze,
            request.text.strip(),
            request.speaker_hint,
            request.listener_hint,
            request.extra_context.strip(),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"語境分析失敗: {exc}") from exc
