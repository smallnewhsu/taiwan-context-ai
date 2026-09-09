"""T22-T50 分語言提示優化評測（不修改模型權重與既有基準結果）。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


LANGUAGE_BY_ID = {
    "T22": "國語", "T23": "台語", "T24": "國台混合", "T25": "國語",
    "T26": "台語", "T27": "國語", "T28": "台語", "T29": "國台混合",
    "T30": "中英混合", "T31": "國語", "T32": "台語", "T33": "客語（四縣）",
    "T34": "國台混合", "T35": "國語", "T36": "中英混合", "T37": "台語",
    "T38": "國語", "T39": "客語（四縣）", "T40": "國台混合", "T41": "台語",
    "T42": "國語", "T43": "中英混合", "T44": "客語（四縣）", "T45": "國台混合",
    "T46": "台語", "T47": "國語", "T48": "中英混合", "T49": "客語（四縣）",
    "T50": "國台混合",
}

PROMPT_BY_LANGUAGE = {
    "國語": "",
    "中英混合": "",
    "台語": (
        "這段錄音主要使用臺灣台語。請完全依照實際音訊逐字轉錄，"
        "保留原本語言，不翻譯、不改寫；聽不清楚時不要根據提示補字。"
    ),
    "國台混合": (
        "這段錄音混合使用國語與臺灣台語。請完全依照實際音訊逐字轉錄，"
        "保留語言切換，不翻譯、不改寫；聽不清楚時不要根據提示補字。"
    ),
    "客語（四縣）": (
        "這段錄音主要使用臺灣客語四縣腔。請完全依照實際音訊逐字轉錄，"
        "保留原本語言，不翻譯、不改寫；聽不清楚時不要根據提示補字。"
    ),
}


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_asr_module(asr_dir: Path):
    sys.path.insert(0, str(asr_dir))
    path = asr_dir / "evaluate_asr.py"
    spec = importlib.util.spec_from_file_location("project_evaluate_asr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_audio(folder: Path, case_id: str) -> Path:
    matches = []
    for suffix in (".wav", ".mp3", ".flac", ".m4a", ".aac"):
        matches.extend(folder.glob(f"{case_id}{suffix}"))
        matches.extend(folder.glob(f"{case_id}{suffix.upper()}"))
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"{case_id} 應有且僅有一個音檔，目前找到 {len(unique)} 個")
    return unique[0]


def validate(folder: Path, baseline_path: Path) -> list[tuple[str, Path, Path]]:
    if not baseline_path.is_file():
        raise RuntimeError(f"找不到基準結果: {baseline_path}")
    samples = []
    for case_id in LANGUAGE_BY_ID:
        audio = find_audio(folder, case_id)
        reference = folder / f"{case_id}_reference.txt"
        if not reference.is_file() or not reference.read_text(encoding="utf-8-sig").strip():
            raise RuntimeError(f"找不到或內容空白: {reference}")
        samples.append((case_id, audio, reference))
    print("資料檢查通過：29 個音檔、29 個標準逐字稿及基準結果。")
    return samples


def normalized_text(asr_module, text: str) -> str:
    return asr_module.remove_special_characters_by_dataset_name(
        asr_module.chinese_number_to_arabic(
            asr_module.s2tw.convert(asr_module.replace_words(text))
        )
    ).lower()


def cer_payload(result) -> dict:
    return {
        "correct_rate": result.correct_rate,
        "cer_rate": result.cer_rate,
        "total_errors": result.total_errors,
        "substitutions_count": result.substitutions_count,
        "deletions_count": result.deletions_count,
        "insertions_count": result.insertions_count,
        "total_chars": result.total_chars,
        "substitutions_errors": result.substitutions_errors,
        "deletions_errors": result.deletions_errors,
        "insertions_errors": result.insertions_errors,
        "reference_highlighted": result.reference_highlighted,
        "hypothesis_highlighted": result.hypothesis_highlighted,
    }


def load_baseline(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("detailed_results") or payload.get("results") or []
    answer = {}
    for row in rows:
        name = row.get("audio_file") or row.get("file_name") or row.get("filename") or ""
        case_id = Path(name).stem.upper()
        if case_id in LANGUAGE_BY_ID:
            answer[case_id] = row
    missing = sorted(set(LANGUAGE_BY_ID) - set(answer))
    if missing:
        raise RuntimeError(f"基準結果缺少案例: {', '.join(missing)}")
    return answer


def summarize(rows: list[dict]) -> dict:
    cer_rows = [row for row in rows if row.get("cer_result")]
    latencies = [row["latency"] for row in rows if row.get("latency")]
    steady = [item for item in latencies if not item.get("is_warmup_sample")]
    return {
        "total_files": len(rows),
        "files_with_cer": len(cer_rows),
        "average_cer": average([row["cer_result"]["cer_rate"] for row in cer_rows]),
        "average_correct_rate": average([row["cer_result"]["correct_rate"] for row in cer_rows]),
        "total_substitutions": sum(row["cer_result"]["substitutions_count"] for row in cer_rows),
        "total_deletions": sum(row["cer_result"]["deletions_count"] for row in cer_rows),
        "total_insertions": sum(row["cer_result"]["insertions_count"] for row in cer_rows),
        "latency_all_files": {
            "sample_count": len(latencies),
            "average_inference_seconds": average([x["inference_seconds"] for x in latencies]),
            "p95_inference_seconds": percentile([x["inference_seconds"] for x in latencies], 95),
            "average_rtf": average([x["rtf"] for x in latencies]),
        },
        "latency_after_warmup": {
            "sample_count": len(steady),
            "average_inference_seconds": average([x["inference_seconds"] for x in steady]),
            "p95_inference_seconds": percentile([x["inference_seconds"] for x in steady], 95),
            "average_rtf": average([x["rtf"] for x in steady]),
        },
    }


def language_comparison(rows: list[dict], baseline: dict[str, dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["language_type"]].append(row)
    output = []
    for language in ("國語", "台語", "國台混合", "中英混合", "客語（四縣）"):
        items = grouped.get(language, [])
        prompt_cers = [x["cer_result"]["cer_rate"] for x in items if x.get("cer_result")]
        baseline_cers = [
            baseline[x["case_id"]]["cer_result"]["cer_rate"]
            for x in items if baseline[x["case_id"]].get("cer_result")
        ]
        baseline_cer = average(baseline_cers)
        prompt_cer = average(prompt_cers)
        output.append({
            "language_type": language,
            "sample_count": len(items),
            "baseline_macro_cer": baseline_cer,
            "prompt_macro_cer": prompt_cer,
            "absolute_cer_change": prompt_cer - baseline_cer,
            "relative_improvement_rate": (
                (baseline_cer - prompt_cer) / baseline_cer if baseline_cer else 0.0
            ),
            "improved_case_count": sum(
                x["cer_result"]["cer_rate"]
                < baseline[x["case_id"]]["cer_result"]["cer_rate"]
                for x in items
            ),
        })
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="T22-T50 分語言提示優化評測")
    parser.add_argument("--folder", default="evaluation/asr_extended")
    parser.add_argument(
        "--baseline", default="evaluation/asr_extended/asr_extended_results_gpu_29.json"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    folder = (repo_root / args.folder).resolve()
    baseline_path = (repo_root / args.baseline).resolve()
    samples = validate(folder, baseline_path)
    baseline = load_baseline(baseline_path)
    asr_dir = repo_root / "services" / "asr"
    asr_module = load_asr_module(asr_dir)

    model_started = time.perf_counter()
    model = asr_module.WhisperModel(
        str(asr_dir / "models"), device="cuda", compute_type="float16"
    )
    model_load_seconds = time.perf_counter() - model_started
    print(f"模型載入完成：{model_load_seconds:.3f} 秒")

    results = []
    for index, (case_id, audio_path, reference_path) in enumerate(samples, 1):
        language = LANGUAGE_BY_ID[case_id]
        prompt = PROMPT_BY_LANGUAGE[language]
        print(f"[{index:02d}/29] {case_id}｜{language}")
        try:
            audio, sample_rate = asr_module.librosa.load(
                str(audio_path), sr=16000, mono=True
            )
            duration = len(audio) / float(sample_rate)
            started = time.perf_counter()
            segments, _ = model.transcribe(
                audio,
                language="zh",
                word_timestamps=False,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=True,
                initial_prompt=prompt,
            )
            generated = "".join(segment.text for segment in segments)
            inference = time.perf_counter() - started
            hypothesis = normalized_text(asr_module, generated)
            reference = reference_path.read_text(encoding="utf-8-sig").strip()
            cer = asr_module.compare_texts(reference, hypothesis)
            if cer is None:
                raise RuntimeError("CER 計算失敗")
            result = {
                "case_id": case_id,
                "audio_file": audio_path.name,
                "language_type": language,
                "prompt_applied": bool(prompt),
                "initial_prompt": prompt,
                "asr_result": hypothesis,
                "original_transcript": reference,
                "has_original_transcript": True,
                "cer_result": cer_payload(cer),
                "latency": {
                    "audio_duration_seconds": round(duration, 4),
                    "inference_seconds": round(inference, 4),
                    "rtf": round(inference / duration, 4) if duration else 0.0,
                    "is_warmup_sample": index == 1,
                    "device": "cuda",
                    "compute_type": "float16",
                },
            }
            (folder / f"{case_id}_prompt_v2_asr.txt").write_text(hypothesis, encoding="utf-8")
            print(
                f"  baseline={baseline[case_id]['cer_result']['cer_rate']:.4f}, "
                f"prompt={cer.cer_rate:.4f}, time={inference:.3f}s"
            )
        except Exception as exc:
            result = {
                "case_id": case_id,
                "audio_file": audio_path.name,
                "language_type": language,
                "prompt_applied": bool(prompt),
                "initial_prompt": prompt,
                "asr_result": None,
                "original_transcript": reference_path.read_text(encoding="utf-8-sig").strip(),
                "has_original_transcript": True,
                "cer_result": None,
                "error": str(exc),
            }
            print(f"  ERROR: {exc}")
        results.append(result)

    comparisons = []
    for row in results:
        base = baseline[row["case_id"]]
        base_cer = (base.get("cer_result") or {}).get("cer_rate")
        prompt_cer = (row.get("cer_result") or {}).get("cer_rate")
        comparisons.append({
            "case_id": row["case_id"],
            "language_type": row["language_type"],
            "baseline_text": base.get("asr_result", ""),
            "prompt_text": row.get("asr_result", ""),
            "reference_text": row.get("original_transcript", ""),
            "baseline_cer": base_cer,
            "prompt_cer": prompt_cer,
            "absolute_cer_change": (
                prompt_cer - base_cer
                if prompt_cer is not None and base_cer is not None else None
            ),
            "improved": (
                prompt_cer < base_cer
                if prompt_cer is not None and base_cer is not None else False
            ),
        })

    language_rows = language_comparison(results, baseline)
    payload = {
        "suite": "Taiwan Context AI ASR Prompt Optimization v2 T22-T50",
        "optimization_type": "instruction_only_prompt_without_lexical_examples_or_weight_fine_tuning",
        "model_load_seconds": round(model_load_seconds, 4),
        "summary": summarize(results),
        "language_comparison": language_rows,
        "detailed_results": results,
    }
    result_path = folder / "asr_prompt_v2_results_gpu_29.json"
    comparison_path = folder / "asr_prompt_v2_comparison_gpu_29.csv"
    language_path = folder / "asr_prompt_v2_language_comparison_gpu_29.csv"
    latency_path = folder / "asr_prompt_v2_latency_gpu_29.csv"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(comparison_path, comparisons, list(comparisons[0]))
    write_csv(language_path, language_rows, list(language_rows[0]))
    latency_rows = [
        {"case_id": row["case_id"], "language_type": row["language_type"], **row["latency"]}
        for row in results if row.get("latency")
    ]
    write_csv(latency_path, latency_rows, list(latency_rows[0]) if latency_rows else ["case_id"])

    summary = payload["summary"]
    print("\n=== T22-T50 分語言提示優化 v2 評測完成 ===")
    print(f"完成案例: {summary['files_with_cer']}/29")
    print(f"提示後平均 CER: {summary['average_cer']:.4f}")
    steady = summary["latency_after_warmup"]
    print(f"排除暖機後平均推論時間: {steady['average_inference_seconds']:.4f} 秒")
    print(f"排除暖機後 P95: {steady['p95_inference_seconds']:.4f} 秒")
    print("各語言前後比較:")
    for row in language_rows:
        print(
            f"- {row['language_type']}: {row['baseline_macro_cer']:.4f} -> "
            f"{row['prompt_macro_cer']:.4f}, 改善案例 {row['improved_case_count']}/"
            f"{row['sample_count']}"
        )
    print(f"詳細結果: {result_path}")
    print(f"逐筆比較: {comparison_path}")
    print(f"分語言比較: {language_path}")
    print(f"延遲結果: {latency_path}")


if __name__ == "__main__":
    main()
