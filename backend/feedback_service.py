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
            "baseline_text": payload["baseline_text"].strip(),
            "prompt_text": payload["prompt_text"].strip(),
            "selected_source": payload["selected_source"],
            "corrected_text": payload["corrected_text"].strip(),
            "reasons": list(payload.get("reasons", [])),
            "consent_to_dataset": bool(payload.get("consent_to_dataset", False)),
            "review_status": "pending",
            "raw_audio_stored": False,
        }
        serialized = json.dumps(record, ensure_ascii=False)
        with self._write_lock:
            with open(PENDING_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
        return {
            "feedback_id": record["feedback_id"],
            "review_status": record["review_status"],
            "dataset_eligible": record["consent_to_dataset"],
            "raw_audio_stored": False,
        }

    @staticmethod
    def _read_jsonl(path):
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def list_pending(self):
        submissions = self._read_jsonl(PENDING_FILE)
        reviewed_ids = {
            item["feedback_id"] for item in self._read_jsonl(REVIEW_FILE)
        }
        return [
            item for item in submissions if item["feedback_id"] not in reviewed_ids
        ]

    def review(self, feedback_id, decision, reviewer_note=""):
        with self._write_lock:
            submissions = {
                item["feedback_id"]: item for item in self._read_jsonl(PENDING_FILE)
            }
            if feedback_id not in submissions:
                raise KeyError("找不到指定的回饋資料")

            reviewed_ids = {
                item["feedback_id"] for item in self._read_jsonl(REVIEW_FILE)
            }
            if feedback_id in reviewed_ids:
                raise ValueError("此筆資料已完成審核")

            submission = submissions[feedback_id]
            event = {
                "review_id": str(uuid.uuid4()),
                "feedback_id": feedback_id,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "reviewer_note": reviewer_note.strip(),
                "dataset_added": False,
            }

            if decision == "approve" and submission["consent_to_dataset"]:
                approved = {
                    **submission,
                    "review_status": "approved",
                    "reviewed_at": event["reviewed_at"],
                    "reviewer_note": event["reviewer_note"],
                }
                with open(APPROVED_FILE, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(approved, ensure_ascii=False) + "\n")
                event["dataset_added"] = True

            with open(REVIEW_FILE, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event
