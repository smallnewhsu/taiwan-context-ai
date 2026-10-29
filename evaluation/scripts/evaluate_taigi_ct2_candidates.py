#!/usr/bin/env python3
"""Evaluate CTranslate2 Taigi candidates on the frozen T81-T100 holdout."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.context_engine import Candidate, ContextEngine


AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
TAIGI_PROMPT = "以下是台語生活對話，請保留原本的台語用詞。"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument(
        "--test-dir", type=Path,
        default=Path("evaluation/asr_finetune/final_holdout_v1/audio"),
    )
    parser.add_argument("--start", type=int, default=81)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--glossary", type=Path,
        default=Path("datasets/taiwan_context/glossary.json"),
    )
    parser.add_argument(
        "--family-prompt", type=Path,
        default=Path("datasets/taiwan_context/asr_prompt_family.txt"),
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "S"))
    )


def distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def find_file(folder: Path, case_id: str, suffixes: tuple[str, ...]) -> Path:
    matches = [folder / f"{case_id}{suffix}" for suffix in suffixes]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"{case_id}: expected one file, found {len(matches)}")
    return matches[0]


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r'[,"\'。，；「」《》:：\[\]、【】〈〉（）『』…(),`&!?！;]', "", text)
    return re.sub(r"\s+", "", text).strip()


def transcribe(model: WhisperModel, audio: Path, prompt: str) -> dict[str, object]:
    started = time.perf_counter()
    segments, _ = model.transcribe(
        str(audio), language="zh", word_timestamps=False, vad_filter=True,
        beam_size=5, condition_on_previous_text=True, initial_prompt=prompt,
    )
    segments = list(segments)
    elapsed = time.perf_counter() - started
    weights = [max(segment.end - segment.start, 0.001) for segment in segments]
    total = sum(weights)
    confidence = (
        sum(segment.avg_logprob * weight for segment, weight in zip(segments, weights)) / total
        if total else -999.0
    )
    return {
        "text": clean("".join(segment.text for segment in segments)),
        "confidence": float(confidence),
        "seconds": elapsed,
    }


def main() -> None:
    args = arguments()
    family = args.family_prompt.read_text(encoding="utf-8-sig").strip()
    prompt_text = "\n".join(part for part in (TAIGI_PROMPT, family) if part)
    engine = ContextEngine(glossary_path=args.glossary, prompt_tolerance=-0.05)
    model = WhisperModel(str(args.model_path), device="cuda", compute_type=args.compute_type)
    rows = []
    for number in range(args.start, args.end + 1):
        case_id = f"T{number}"
        audio = find_file(args.test_dir, case_id, AUDIO_SUFFIXES)
        reference_path = args.test_dir / f"{case_id}_reference.txt"
        reference = reference_path.read_text(encoding="utf-8-sig").strip()
        baseline = transcribe(model, audio, "")
        prompted = transcribe(model, audio, prompt_text)
        decision = engine.decide(
            Candidate(source="baseline", text=baseline["text"], confidence=baseline["confidence"]),
            Candidate(source="prompt", text=prompted["text"], confidence=prompted["confidence"]),
        )
        ref_norm, hyp_norm = normalize(reference), normalize(decision.selected_text)
        errors = distance(ref_norm, hyp_norm)
        row = {
            "case_id": case_id, "reference": reference,
            "baseline": baseline, "prompt": prompted,
            "selected_source": decision.selected_source,
            "selected_text": decision.selected_text,
            "needs_confirmation": decision.needs_confirmation,
            "reasons": decision.reasons,
            "reference_characters": len(ref_norm), "errors": errors,
            "cer": errors / len(ref_norm) if ref_norm else None,
            "dual_pass_seconds": baseline["seconds"] + prompted["seconds"],
        }
        rows.append(row)
        print(f"{case_id}: CER={row['cer']:.4f} {decision.selected_source} {decision.selected_text}")

    total_errors = sum(row["errors"] for row in rows)
    total_chars = sum(row["reference_characters"] for row in rows)
    summary = {
        "suite": f"T{args.start}-T{args.end} frozen CT2 production-route evaluation",
        "model_path": str(args.model_path), "compute_type": args.compute_type,
        "total_cases": len(rows), "api_error_count": 0,
        "macro_cer": sum(row["cer"] for row in rows) / len(rows),
        "micro_cer": total_errors / total_chars,
        "total_errors": total_errors,
        "total_reference_characters": total_chars,
        "mean_dual_pass_seconds": sum(row["dual_pass_seconds"] for row in rows) / len(rows),
        "needs_confirmation_count": sum(row["needs_confirmation"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== CT2 frozen evaluation complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output.resolve()}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
