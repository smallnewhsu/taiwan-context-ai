from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class Candidate:
    source: str
    text: str
    confidence: float


@dataclass
class ContextDecision:
    selected_source: str
    selected_text: str
    decision_confidence: float
    needs_confirmation: bool
    reasons: List[str] = field(default_factory=list)
    baseline_score: float = 0.0
    prompt_score: float = 0.0

    def to_dict(self):
        return asdict(self)
