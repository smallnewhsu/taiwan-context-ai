#!/usr/bin/env python3
"""Score an existing merged-model verification JSON against frozen references."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Taigi merge verification")
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\taigi_v3_merge_verification.json"
        ),
    )
    parser.add_argument(
        "--validated-results",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\taigi_v3_final_results.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\taigi_v3_merge_verification_scored.json"
        ),
    )
    return parser.parse_args()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def main() -> None:
    args = parse_args()
    verification = json.loads(args.verification.read_text(encoding="utf-8-sig"))
    validated = json.loads(args.validated_results.read_text(encoding="utf-8-sig"))
    reference_by_id = {
        row["case_id"]: {
            "reference": row["reference"],
            "adapter_text": row["adapter_text"],
        }
        for row in validated["results"]
    }
    cases: list[dict[str, object]] = []
    for row in verification["cases"]:
        case_id = row["case_id"]
        source = reference_by_id[case_id]
        reference = normalize(source["reference"])
        adapter_text = normalize(source["adapter_text"])
        merged_text = normalize(row["merged_model_text"])
        adapter_errors = distance(reference, adapter_text)
        merged_errors = distance(reference, merged_text)
        cases.append(
            {
                "case_id": case_id,
                "reference": source["reference"],
                "adapter_text": source["adapter_text"],
                "merged_text": row["merged_model_text"],
                "reference_characters": len(reference),
                "adapter_errors": adapter_errors,
                "merged_errors": merged_errors,
                "adapter_cer": adapter_errors / len(reference),
                "merged_cer": merged_errors / len(reference),
                "merged_minus_adapter_errors": merged_errors - adapter_errors,
            }
        )

    total_characters = sum(int(row["reference_characters"]) for row in cases)
    adapter_errors = sum(int(row["adapter_errors"]) for row in cases)
    merged_errors = sum(int(row["merged_errors"]) for row in cases)
    summary = {
        "suite": "Taigi LoRA v3 merged model scored verification",
        "total_cases": len(cases),
        "exact_match_count": int(verification["exact_match_count"]),
        "mismatch_count": int(verification["mismatch_count"]),
        "adapter_micro_cer": adapter_errors / total_characters,
        "merged_micro_cer": merged_errors / total_characters,
        "merged_minus_adapter_micro_cer": (
            merged_errors - adapter_errors
        )
        / total_characters,
        "adapter_total_errors": adapter_errors,
        "merged_total_errors": merged_errors,
        "merged_improved_cases": sum(
            1 for row in cases if int(row["merged_errors"]) < int(row["adapter_errors"])
        ),
        "merged_unchanged_cases": sum(
            1 for row in cases if int(row["merged_errors"]) == int(row["adapter_errors"])
        ),
        "merged_regressed_cases": sum(
            1 for row in cases if int(row["merged_errors"]) > int(row["adapter_errors"])
        ),
        "mean_processing_seconds": verification.get("mean_processing_seconds"),
    }
    result = {"summary": summary, "cases": cases}
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Scored merge verification complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
