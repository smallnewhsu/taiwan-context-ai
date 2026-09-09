import json
import threading
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTION_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "taiwan_context"
    / "feedback"
    / "approved_corrections.jsonl"
)


class CorrectionMemory:
    """Local, human-reviewed overrides for the exact same audio bytes."""

    def __init__(self):
        self._lock = threading.Lock()

    @staticmethod
    def _read_records():
        if not CORRECTION_FILE.exists():
            return []
        records = []
        with open(CORRECTION_FILE, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("audio_sha256") and row.get("corrected_text"):
                    records.append(row)
        return records

    def find(self, audio_sha256):
        # Last approved correction wins, so a later review can supersede it.
        for row in reversed(self._read_records()):
            if row.get("audio_sha256") == audio_sha256:
                return row
        return None

    def approve(self, submission, review_event):
        audio_sha256 = str(submission.get("audio_sha256", "")).strip().lower()
        corrected_text = str(submission.get("corrected_text", "")).strip()
        if len(audio_sha256) != 64 or not corrected_text:
            return None
        record = {
            "audio_sha256": audio_sha256,
            "corrected_text": corrected_text,
            "feedback_id": submission.get("feedback_id"),
            "review_id": review_event.get("review_id"),
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "source": "human_review",
            "raw_audio_stored": False,
        }
        context_payload = None
        if submission.get("human_context"):
            context_payload = {
                "ai_context": submission.get("ai_context") or {},
                "human_context": submission.get("human_context") or {},
            }
        else:
            for reason in submission.get("reasons", []):
                if not str(reason).startswith("context_edit_json:"):
                    continue
                try:
                    context_payload = json.loads(str(reason)[len("context_edit_json:"):])
                except json.JSONDecodeError:
                    context_payload = None
                break
        if context_payload and context_payload.get("human_context"):
            merged_context = {
                **dict(context_payload.get("ai_context") or {}),
                **dict(context_payload["human_context"]),
            }
            if merged_context.get("possible_intent"):
                merged_context["possible_intents"] = [str(merged_context["possible_intent"])]
            if isinstance(merged_context.get("basis"), str):
                merged_context["basis"] = [merged_context["basis"]]
            record["human_context"] = merged_context
        CORRECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(CORRECTION_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
