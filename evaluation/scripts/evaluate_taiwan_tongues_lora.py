#!/usr/bin/env python3
"""Compare a Taiwan Tongues base model and LoRA adapter on a frozen test set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Taiwan Tongues INT8 base vs LoRA adapter"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\models\Taiwan-Tongues-ASR-CE-pretrained-v1.0"
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path(r"C:\taiwan-context-training\adapters\taigi_pilot_lora"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(r"C:\taiwan-context-ai\evaluation\asr_multilingual"),
    )
    parser.add_argument("--start", type=int, default=51)
    parser.add_argument("--end", type=int, default=60)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_multilingual\taigi_lora_pilot"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """Return substitutions, deletions, insertions using Levenshtein DP."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for i in range(1, rows):
        table[i][0] = (i, 0, i, 0)
    for j in range(1, columns):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, columns):
            if reference[i - 1] == hypothesis[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            deletion = table[i - 1][j]
            insertion = table[i][j - 1]
            substitution = table[i - 1][j - 1]
            choices = [
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
                (
                    substitution[0] + 1,
                    substitution[1] + 1,
                    substitution[2],
                    substitution[3],
                ),
            ]
            table[i][j] = min(choices, key=lambda item: item[0])
    _, substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions


def find_audio(folder: Path, case_id: str) -> Path:
    matches = [folder / f"{case_id}{suffix}" for suffix in AUDIO_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one audio file for {case_id}; found {len(matches)}"
        )
    return matches[0]


def transcribe(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    audio_path: Path,
    max_new_tokens: int,
) -> tuple[str, float]:
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    features = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda:0")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        generated = model.generate(
            input_features=features,
            language="zh",
            task="transcribe",
            max_new_tokens=max_new_tokens,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return text, elapsed


def score(reference: str, hypothesis: str) -> dict[str, object]:
    normalized_reference = normalize_for_cer(reference)
    normalized_hypothesis = normalize_for_cer(hypothesis)
    substitutions, deletions, insertions = edit_counts(
        normalized_reference, normalized_hypothesis
    )
    errors = substitutions + deletions + insertions
    length = len(normalized_reference)
    return {
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "reference_characters": length,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "cer": errors / length if length else None,
    }


def aggregate(results: list[dict[str, object]], prefix: str) -> dict[str, object]:
    scores = [row[f"{prefix}_score"] for row in results]
    valid = [value for value in scores if value["cer"] is not None]
    total_errors = sum(int(value["errors"]) for value in valid)
    total_chars = sum(int(value["reference_characters"]) for value in valid)
    times = [float(row[f"{prefix}_seconds"]) for row in results]
    return {
        "macro_cer": sum(float(value["cer"]) for value in valid) / len(valid),
        "micro_cer": total_errors / total_chars if total_chars else None,
        "total_errors": total_errors,
        "total_reference_characters": total_chars,
        "mean_seconds": sum(times) / len(times),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter not found: {args.adapter_path}")

    cases: list[dict[str, object]] = []
    for number in range(args.start, args.end + 1):
        case_id = f"T{number}"
        reference_path = args.test_dir / f"{case_id}_reference.txt"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference not found: {reference_path}")
        cases.append(
            {
                "case_id": case_id,
                "audio_path": find_audio(args.test_dir, case_id),
                "reference": reference_path.read_text(encoding="utf-8-sig").strip(),
            }
        )

    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language="zh",
        task="transcribe",
        local_files_only=True,
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    results: list[dict[str, object]] = []
    print(f"Evaluating base model on {len(cases)} frozen cases...")
    for case in cases:
        hypothesis, elapsed = transcribe(
            base_model, processor, case["audio_path"], args.max_new_tokens
        )
        result = dict(case)
        result["audio_path"] = str(case["audio_path"])
        result["base_text"] = hypothesis
        result["base_seconds"] = elapsed
        result["base_score"] = score(str(case["reference"]), hypothesis)
        results.append(result)
        print(
            f"{case['case_id']} base CER={result['base_score']['cer']:.4f}: {hypothesis}"
        )

    adapter_model = PeftModel.from_pretrained(
        base_model, args.adapter_path, is_trainable=False
    ).eval()
    print(f"Evaluating adapter on {len(cases)} frozen cases...")
    for result in results:
        hypothesis, elapsed = transcribe(
            adapter_model,
            processor,
            Path(str(result["audio_path"])),
            args.max_new_tokens,
        )
        result["adapter_text"] = hypothesis
        result["adapter_seconds"] = elapsed
        result["adapter_score"] = score(str(result["reference"]), hypothesis)
        result["cer_delta"] = (
            float(result["adapter_score"]["cer"])
            - float(result["base_score"]["cer"])
        )
        print(
            f"{result['case_id']} adapter CER={result['adapter_score']['cer']:.4f}: "
            f"{hypothesis}"
        )

    base_summary = aggregate(results, "base")
    adapter_summary = aggregate(results, "adapter")
    base_micro = float(base_summary["micro_cer"])
    adapter_micro = float(adapter_summary["micro_cer"])
    relative_improvement = (
        (base_micro - adapter_micro) / base_micro if base_micro else None
    )
    summary = {
        "suite": f"T{args.start}-T{args.end} frozen LoRA comparison",
        "total_cases": len(results),
        "base": base_summary,
        "adapter": adapter_summary,
        "micro_cer_absolute_change": adapter_micro - base_micro,
        "micro_cer_relative_improvement": relative_improvement,
        "improved_cases": sum(1 for row in results if row["cer_delta"] < 0),
        "unchanged_cases": sum(1 for row in results if row["cer_delta"] == 0),
        "regressed_cases": sum(1 for row in results if row["cer_delta"] > 0),
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_name(args.output_prefix.name + "_results.json")
    csv_path = args.output_prefix.with_name(args.output_prefix.name + "_comparison.csv")
    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "reference",
                "base_text",
                "base_cer",
                "adapter_text",
                "adapter_cer",
                "cer_delta",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["case_id"],
                    row["reference"],
                    row["base_text"],
                    row["base_score"]["cer"],
                    row["adapter_text"],
                    row["adapter_score"]["cer"],
                    row["cer_delta"],
                ]
            )

    print("=== Frozen comparison complete ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {json_path}")
    print(f"Comparison: {csv_path}")


if __name__ == "__main__":
    main()
