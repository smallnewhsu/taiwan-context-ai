import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_DIR = PROJECT_ROOT / "datasets" / "taiwan_context" / "feedback"
PENDING_FILE = FEEDBACK_DIR / "pending_feedback.jsonl"


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
