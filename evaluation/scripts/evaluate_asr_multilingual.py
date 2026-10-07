"""T51-T80 multilingual ASR evaluation (Taigi, Hakka, Vietnamese).

Run from the repository root:
    python evaluation/scripts/evaluate_asr_multilingual.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import librosa
from faster_whisper import WhisperModel


GROUPS = {
    **{f"T{i}": ("台語", "zh") for i in range(51, 61)},
    **{f"T{i}": ("客語", "zh") for i in range(61, 71)},
    **{f"T{i}": ("越南語", "vi") for i in range(71, 81)},
}

PROMPTS = {
    "台語": "以下是臺灣台語生活對話，請保留人名、時間、金額、否定與限制語氣。",
    "客語": "以下是臺灣客語四縣腔生活對話，請保留人名、時間、金額、否定與限制語氣。",
    "越南語": "Đây là hội thoại tiếng Việt trong đời sống hằng ngày. Hãy giữ nguyên tên, thời gian, số tiền và ý phủ định.",
}

AUDIO_EXTENSIONS = (".m4a", ".wav", ".mp3", ".flac", ".aac")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    return "".join(
        char for char in text
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    ref, hyp = list(reference), list(hypothesis)
    rows, cols = len(ref) + 1, len(hyp) + 1
    dp = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            choices = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            dp[i][j] = min(choices, key=lambda item: item[0])
    _, substitutions, deletions, insertions = dp[-1][-1]
    return substitutions, deletions, insertions


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def locate_audio(folder: Path, case_id: str) -> Path | None:
    matches = [folder / f"{case_id}{ext}" for ext in AUDIO_EXTENSIONS]
    matches = [path for path in matches if path.is_file()]
    if len(matches) > 1:
        raise SystemExit(f"{case_id}有多個音檔: {', '.join(str(x) for x in matches)}")
    return matches[0] if matches else None


def validate(folder: Path) -> list[tuple[str, Path, Path]]:
    records = []
    errors = []
    for case_id in GROUPS:
        audio = locate_audio(folder, case_id)
        reference = folder / f"{case_id}_reference.txt"
        if audio is None:
            errors.append(f"缺少音檔: {case_id}")
        if not reference.is_file():
            errors.append(f"缺少逐字稿: {reference.name}")
        elif not reference.read_text(encoding="utf-8-sig").strip():
            errors.append(f"逐字稿為空白: {reference.name}")
        if audio is not None and reference.is_file():
            records.append((case_id, audio, reference))
    if errors:
        raise SystemExit("\n".join(errors))
    print("資料檢查通過：30個音檔與30個標準逐字稿。")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="T51-T80多語ASR GPU評測")
    parser.add_argument("--folder", default="evaluation/asr_multilingual")
    parser.add_argument("--model-path", default="services/asr/models")
    parser.add_argument("--output-prefix", default="asr_multilingual_gpu_30")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    folder = (root / args.folder).resolve()
    model_path = (root / args.model_path).resolve()
    if not folder.is_dir():
        raise SystemExit(f"找不到評測資料夾: {folder}")
    if not model_path.exists():
        raise SystemExit(f"找不到ASR模型: {model_path}")

    records = validate(folder)
    load_started = time.perf_counter()
    model = WhisperModel(str(model_path), device=args.device, compute_type=args.compute_type)
    model_load_seconds = time.perf_counter() - load_started

    results = []
    for index, (case_id, audio_path, reference_path) in enumerate(records, 1):
        language_label, language_code = GROUPS[case_id]
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / float(sample_rate)
        started = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            language=language_code,
            initial_prompt=PROMPTS[language_label],
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
            word_timestamps=False,
        )
        hypothesis_raw = "".join(segment.text for segment in segments).strip()
        inference_seconds = time.perf_counter() - started
        reference_raw = reference_path.read_text(encoding="utf-8-sig").strip()
        reference = normalize(reference_raw)
        hypothesis = normalize(hypothesis_raw)
        substitutions, deletions, insertions = edit_counts(reference, hypothesis)
        errors = substitutions + deletions + insertions
        cer = errors / len(reference) if reference else None
        item = {
            "case_id": case_id,
            "audio_file": audio_path.name,
            "language": language_label,
            "language_code": language_code,
            "reference": reference_raw,
            "hypothesis": hypothesis_raw,
            "normalized_reference": reference,
            "normalized_hypothesis": hypothesis,
            "cer": cer,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "errors": errors,
            "reference_characters": len(reference),
            "audio_duration_seconds": round(duration, 4),
            "inference_seconds": round(inference_seconds, 4),
            "rtf": round(inference_seconds / duration, 4) if duration else None,
            "detected_language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "is_warmup_sample": index == 1,
        }
        results.append(item)
        print(f"[{index:02d}/30] {case_id} {language_label} CER={cer:.4f}  {hypothesis_raw}")

    grouped = defaultdict(list)
    for item in results:
        grouped[item["language"]].append(item)

    language_summary = {}
    for language, items in grouped.items():
        total_errors = sum(item["errors"] for item in items)
        total_chars = sum(item["reference_characters"] for item in items)
        cers = [item["cer"] for item in items if item["cer"] is not None]
        times = [item["inference_seconds"] for item in items]
        language_summary[language] = {
            "sample_count": len(items),
            "macro_cer": statistics.fmean(cers),
            "micro_cer": total_errors / total_chars if total_chars else None,
            "mean_inference_seconds": statistics.fmean(times),
            "p95_inference_seconds": percentile(times, 0.95),
        }

    cers = [item["cer"] for item in results if item["cer"] is not None]
    steady_times = [item["inference_seconds"] for item in results[1:]]
    summary = {
        "suite": "Taiwan Context AI ASR Multilingual T51-T80",
        "total_cases": len(results),
        "completed_cases": len(cers),
        "model_path": str(model_path),
        "model_load_seconds": round(model_load_seconds, 4),
        "macro_average_cer": statistics.fmean(cers),
        "total_errors": sum(item["errors"] for item in results),
        "total_reference_characters": sum(item["reference_characters"] for item in results),
        "micro_cer": sum(item["errors"] for item in results) / sum(item["reference_characters"] for item in results),
        "warmup_processing_seconds": results[0]["inference_seconds"],
        "steady_state_sample_count": len(steady_times),
        "steady_state_mean_processing_seconds": statistics.fmean(steady_times),
        "steady_state_median_processing_seconds": statistics.median(steady_times),
        "steady_state_p95_processing_seconds": percentile(steady_times, 0.95),
    }
    payload = {"summary": summary, "language_summary": language_summary, "results": results}

    output_json = folder / f"{args.output_prefix}_results.json"
    case_csv = folder / f"{args.output_prefix}_comparison.csv"
    language_csv = folder / f"{args.output_prefix}_language_summary.csv"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with case_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["case_id", "language", "audio_file", "reference", "hypothesis", "cer", "substitutions", "deletions", "insertions", "audio_duration_seconds", "inference_seconds", "rtf"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({key: item.get(key) for key in fields})

    with language_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["language", "sample_count", "macro_cer", "micro_cer", "mean_inference_seconds", "p95_inference_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for language, values in language_summary.items():
            writer.writerow({"language": language, **values})

    print("\n=== T51-T80多語評測完成 ===")
    print(f"總案例數: {summary['total_cases']}")
    print(f"整體Macro CER: {summary['macro_average_cer']:.4f}")
    print(f"整體Micro CER: {summary['micro_cer']:.4f}")
    print(f"穩態平均推論時間: {summary['steady_state_mean_processing_seconds']:.4f}秒")
    print(f"穩態P95: {summary['steady_state_p95_processing_seconds']:.4f}秒")
    for language, values in language_summary.items():
        print(f"- {language}: n={values['sample_count']}, macro={values['macro_cer']:.4f}, micro={values['micro_cer']:.4f}")
    print(f"詳細結果: {output_json}")
    print(f"逐筆比較: {case_csv}")
    print(f"分語言摘要: {language_csv}")


if __name__ == "__main__":
    main()
