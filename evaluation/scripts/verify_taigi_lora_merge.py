#!/usr/bin/env python3
"""Verify that the merged HF model reproduces the validated PEFT outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify merged Taigi LoRA model")
    parser.add_argument(
        "--merged-model",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\models\taigi_lora_v3_merged_hf"
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\audio"
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
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\taigi_v3_merge_verification.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    validated = json.loads(args.validated_results.read_text(encoding="utf-8-sig"))
    expected = {
        row["case_id"]: row["adapter_text"] for row in validated["results"]
    }
    if len(expected) != 20:
        raise ValueError(f"Expected 20 validated cases; found {len(expected)}")

    processor = WhisperProcessor.from_pretrained(
        args.merged_model,
        language="zh",
        task="transcribe",
        local_files_only=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.merged_model,
        dtype=torch.float16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    cases: list[dict[str, object]] = []
    for number in range(81, 101):
        case_id = f"T{number}"
        matches = [
            path
            for suffix in (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
            if (path := args.test_dir / f"{case_id}{suffix}").is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"{case_id}: expected one audio file")
        audio, _ = librosa.load(matches[0], sr=16000, mono=True)
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
                max_new_tokens=128,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        matches_expected = actual == expected[case_id]
        cases.append(
            {
                "case_id": case_id,
                "expected_adapter_text": expected[case_id],
                "merged_model_text": actual,
                "exact_match": matches_expected,
                "processing_seconds": elapsed,
            }
        )
        print(f"{case_id}: {'MATCH' if matches_expected else 'MISMATCH'}")

    mismatches = [case for case in cases if not case["exact_match"]]
    report = {
        "suite": "Taigi LoRA v3 PEFT-to-merged verification",
        "total_cases": len(cases),
        "exact_match_count": len(cases) - len(mismatches),
        "mismatch_count": len(mismatches),
        "all_outputs_identical": not mismatches,
        "mean_processing_seconds": sum(
            float(case["processing_seconds"]) for case in cases
        )
        / len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Merge verification complete ===")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")
    if mismatches:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
