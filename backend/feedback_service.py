import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback"
PENDING_FILE = FEEDBACK_DIR / "pending_feedback.jsonl"
REVIEW_FILE = FEEDBACK_DIR / "review_events.jsonl"
APPROVED_FILE = FEEDBACK_DIR / "approved_feedback.jsonl"


class FeedbackService:
    def __init__(self):
        self._write_lock = threading.Lock()

    def save(self, payload):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "feedback_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audio_file": Path(payload["audio_file"]).name,
            "audio_sha256": str(payload.get("audio_sha256", "")).lower(),
            "baseline_text": payload["baseline_text"].strip(),
            "prompt_text": payload["prompt_text"].strip(),
            "selected_source": payload["selected_source"],
            "corrected_text": payload["corrected_text"].strip(),
            "reasons": list(payload.get("reasons", [])),
            "consent_to_dataset": bool(payload.get("consent_to_dataset", False)),
            "review_status": "pending",
            "raw_audio_stored": False,
        }
        if payload.get("feedback_type") == "speech_context":
            record["feedback_type"] = "speech_context"
            record["ai_context"] = dict(payload.get("ai_context") or {})
            record["human_context"] = dict(payload.get("human_context") or {})
            required = ("literal_meaning", "possible_intent", "basis", "suggested_reply")
            if any(not str(record["human_context"].get(key, "")).strip() for key in required):
                raise ValueError("四個修改欄位都必須有內容")
        with self._write_lock:
            with open(PENDING_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "feedback_id": record["feedback_id"],
            "review_status": "pending",
            "dataset_eligible": record["consent_to_dataset"],
            "correction_memory_eligible": len(record["audio_sha256"]) == 64,
            "raw_audio_stored": False,
        }

    def save_context(self, payload):
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        transcription = payload["transcription"].strip()
        record = {
            "feedback_id": str(uuid.uuid4()),
            "feedback_type": "speech_context",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audio_file": Path(payload["audio_file"]).name,
            "audio_sha256": str(payload["audio_sha256"]).lower(),
            "baseline_text": transcription,
            "prompt_text": transcription,
            "selected_source": "manual",
            "corrected_text": transcription,
            "reasons": ["human_context_edit"],
            "ai_context": dict(payload.get("ai_context") or {}),
            "human_context": dict(payload.get("human_context") or {}),
            "consent_to_dataset": bool(payload.get("consent_to_dataset", False)),
            "review_status": "pending",
            "raw_audio_stored": False,
        }
        required = ("literal_meaning", "possible_intent", "basis", "suggested_reply")
        if any(not str(record["human_context"].get(key, "")).strip() for key in required):
            raise ValueError("四個修改欄位都必須有內容")
        with self._write_lock:
            with open(PENDING_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "feedback_id": record["feedback_id"],
            "review_status": "pending",
            "dataset_eligible": record["consent_to_dataset"],
            "context_memory_pending": True,
            "raw_audio_stored": False,
        }

    @staticmethod
    def _read_jsonl(path):
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def get_submission(self, feedback_id):
        for item in reversed(self._read_jsonl(PENDING_FILE)):
            if item.get("feedback_id") == feedback_id:
                return item
        raise KeyError("找不到指定的回饋資料")

    def list_pending(self):
        reviewed = {item["feedback_id"] for item in self._read_jsonl(REVIEW_FILE)}
        return [item for item in self._read_jsonl(PENDING_FILE) if item["feedback_id"] not in reviewed]

    def review(self, feedback_id, decision, reviewer_note=""):
        with self._write_lock:
            submission = self.get_submission(feedback_id)
            reviewed = {item["feedback_id"] for item in self._read_jsonl(REVIEW_FILE)}
            if feedback_id in reviewed:
                raise ValueError("此筆資料已完成審核")
            event = {
                "review_id": str(uuid.uuid4()),
                "feedback_id": feedback_id,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "reviewer_note": reviewer_note.strip(),
                "dataset_added": False,
                "correction_memory_added": False,
            }
            if decision == "approve" and submission["consent_to_dataset"]:
                approved = {**submission, "review_status": "approved", "reviewed_at": event["reviewed_at"], "reviewer_note": event["reviewer_note"]}
                with open(APPROVED_FILE, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(approved, ensure_ascii=False) + "\n")
                event["dataset_added"] = True
            with open(REVIEW_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
