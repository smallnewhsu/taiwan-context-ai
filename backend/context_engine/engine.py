import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .schemas import Candidate, ContextDecision


class ContextEngine:
    """Conservative candidate reranker for Taiwan-context ASR.

    The engine never reads the reference transcript. It selects between ASR
    candidates and asks for confirmation when both candidates are unreliable.
    """

    def __init__(
        self,
        glossary_path=None,
        prompt_tolerance=-0.05,
        low_confidence_threshold=-0.35,
        ambiguity_threshold=0.03,
        large_difference_threshold=0.45,
        glossary_bonus=0.02,
    ):
        self.prompt_tolerance = prompt_tolerance
        self.low_confidence_threshold = low_confidence_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.large_difference_threshold = large_difference_threshold
        self.glossary_bonus = glossary_bonus
        self.terms = []
        if glossary_path:
            with open(glossary_path, "r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
            self.terms = [item["term"] for item in payload.get("entries", [])]

    @staticmethod
    def _normalize(text):
        return re.sub(r"\s+", "", (text or "").strip())

    def _term_hits(self, text):
        normalized = self._normalize(text)
        return sum(1 for term in self.terms if term and term in normalized)

    @staticmethod
    def _repetition_penalty(text):
        normalized = re.sub(r"\s+", "", text or "")
        if len(normalized) < 4:
            return 0.0
        repeated = sum(
            1 for index in range(len(normalized) - 1)
            if normalized[index] == normalized[index + 1]
        )
        return min(0.06, repeated * 0.015)

    def _score(self, candidate):
        return (
            candidate.confidence
            + self._term_hits(candidate.text) * self.glossary_bonus
            - self._repetition_penalty(candidate.text)
        )

    def decide(self, baseline: Candidate, prompt: Candidate):
        baseline_text = self._normalize(baseline.text)
        prompt_text = self._normalize(prompt.text)
        baseline_score = self._score(baseline)
        prompt_score = self._score(prompt)

        if baseline_text == prompt_text:
            return ContextDecision(
                selected_source="baseline",
                selected_text=baseline.text,
                decision_confidence=1.0,
                needs_confirmation=False,
                reasons=["identical_candidates"],
                baseline_score=baseline_score,
                prompt_score=prompt_score,
            )

        # Preserve the independently validated -0.05 prompt tolerance.
        use_prompt = prompt_score > baseline_score + self.prompt_tolerance
        selected = prompt if use_prompt else baseline
        score_gap = abs(prompt_score - baseline_score)
        maximum_confidence = max(baseline.confidence, prompt.confidence)
        similarity = SequenceMatcher(None, baseline_text, prompt_text).ratio()

        reasons = ["prompt_selected" if use_prompt else "baseline_selected"]
        needs_confirmation = False

        if maximum_confidence < self.low_confidence_threshold:
            needs_confirmation = True
            reasons.append("both_candidates_low_confidence")
        if score_gap < self.ambiguity_threshold and similarity < 0.85:
            needs_confirmation = True
            reasons.append("ambiguous_candidates")
        if 1.0 - similarity > self.large_difference_threshold and score_gap < 0.08:
            needs_confirmation = True
            reasons.append("candidates_disagree")

        decision_confidence = max(0.0, min(1.0, 0.5 + score_gap))
        return ContextDecision(
            selected_source=selected.source,
            selected_text=selected.text,
            decision_confidence=decision_confidence,
            needs_confirmation=needs_confirmation,
            reasons=reasons,
            baseline_score=baseline_score,
            prompt_score=prompt_score,
        )
