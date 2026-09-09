"""由既有 baseline 與 prompt v2 結果建立 T22-T50 語言條件路由 v3。"""

from __future__ import annotations

import csv
import copy
import json
from collections import defaultdict
from pathlib import Path


PROMPT_LANGUAGES = {"國台混合", "客語（四縣）"}
LANGUAGE_ORDER = ["國語", "台語", "國台混合", "中英混合", "客語（四縣）"]


def load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("detailed_results") or payload.get("results") or []
    indexed = {}
    for row in rows:
        name = row.get("case_id") or Path(
            row.get("audio_file") or row.get("file_name") or row.get("filename") or ""
        ).stem.upper()
        if name:
            indexed[name] = row
    return payload, indexed


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    folder = repo_root / "evaluation" / "asr_extended"
    baseline_path = folder / "asr_extended_results_gpu_29.json"
    prompt_path = folder / "asr_prompt_v2_results_gpu_29.json"
    if not baseline_path.is_file() or not prompt_path.is_file():
        raise SystemExit(
            "缺少輸入檔，請確認 baseline 與 prompt v2 JSON 都位於 evaluation/asr_extended。"
        )

    baseline_payload, baseline = load_rows(baseline_path)
    prompt_payload, prompt = load_rows(prompt_path)
    expected = [f"T{number:02d}" for number in range(22, 51)]
    missing = [case_id for case_id in expected if case_id not in baseline or case_id not in prompt]
    if missing:
        raise SystemExit(f"結果缺少案例: {', '.join(missing)}")

    routed = []
    comparison = []
    for case_id in expected:
        base = baseline[case_id]
        candidate = prompt[case_id]
        language = candidate.get("language_type") or base.get("language_type")
        use_prompt = language in PROMPT_LANGUAGES
        selected = copy.deepcopy(candidate if use_prompt else base)
        selected["case_id"] = case_id
        selected["language_type"] = language
        selected["selected_source"] = "prompt_v2" if use_prompt else "baseline"
        routed.append(selected)

        base_cer = (base.get("cer_result") or {}).get("cer_rate")
        prompt_cer = (candidate.get("cer_result") or {}).get("cer_rate")
        selected_cer = (selected.get("cer_result") or {}).get("cer_rate")
        comparison.append({
            "case_id": case_id,
            "language_type": language,
            "selected_source": selected["selected_source"],
            "baseline_cer": base_cer,
            "prompt_v2_cer": prompt_cer,
            "hybrid_v3_cer": selected_cer,
            "cer_change_vs_baseline": (
                selected_cer - base_cer
                if selected_cer is not None and base_cer is not None else None
            ),
            "baseline_text": base.get("asr_result", ""),
            "prompt_v2_text": candidate.get("asr_result", ""),
            "selected_text": selected.get("asr_result", ""),
            "reference_text": selected.get("original_transcript", ""),
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in comparison:
        grouped[row["language_type"]].append(row)
    language_summary = []
    for language in LANGUAGE_ORDER:
        rows = grouped[language]
        baseline_cer = average([float(row["baseline_cer"]) for row in rows])
        hybrid_cer = average([float(row["hybrid_v3_cer"]) for row in rows])
        language_summary.append({
            "language_type": language,
            "sample_count": len(rows),
            "selected_source": "prompt_v2" if language in PROMPT_LANGUAGES else "baseline",
            "baseline_macro_cer": baseline_cer,
            "hybrid_v3_macro_cer": hybrid_cer,
            "absolute_cer_change": hybrid_cer - baseline_cer,
            "relative_improvement_rate": (
                (baseline_cer - hybrid_cer) / baseline_cer if baseline_cer else 0.0
            ),
        })

    cer_rows = [row for row in comparison if row["hybrid_v3_cer"] is not None]
    baseline_average = average([float(row["baseline_cer"]) for row in cer_rows])
    hybrid_average = average([float(row["hybrid_v3_cer"]) for row in cer_rows])
    payload = {
        "suite": "Taiwan Context AI ASR Language-Conditional Hybrid v3 T22-T50",
        "strategy": {
            "baseline_languages": ["國語", "台語", "中英混合"],
            "prompt_v2_languages": ["國台混合", "客語（四縣）"],
            "weight_fine_tuning_used": False,
        },
        "summary": {
            "total_files": len(routed),
            "files_with_cer": len(cer_rows),
            "baseline_average_cer": baseline_average,
            "hybrid_v3_average_cer": hybrid_average,
            "absolute_cer_change": hybrid_average - baseline_average,
            "relative_improvement_rate": (
                (baseline_average - hybrid_average) / baseline_average
                if baseline_average else 0.0
            ),
            "latency_note": (
                "baseline evaluator did not record latency; prompt v2 steady-state latency "
                "is retained as separate benchmark evidence"
            ),
            "prompt_v2_latency_after_warmup": (
                prompt_payload.get("summary", {}).get("latency_after_warmup", {})
            ),
        },
        "language_summary": language_summary,
        "detailed_results": routed,
    }

    json_path = folder / "asr_hybrid_v3_results_gpu_29.json"
    detail_path = folder / "asr_hybrid_v3_comparison_gpu_29.csv"
    language_path = folder / "asr_hybrid_v3_language_summary_gpu_29.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with detail_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with language_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(language_summary[0]))
        writer.writeheader()
        writer.writerows(language_summary)

    print("=== T22-T50 語言條件路由 v3 完成 ===")
    print(f"案例數: {len(routed)}")
    print(f"基準平均 CER: {baseline_average:.4f}")
    print(f"v3 平均 CER: {hybrid_average:.4f}")
    print(f"相對改善率: {payload['summary']['relative_improvement_rate']:.2%}")
    for row in language_summary:
        print(
            f"- {row['language_type']}: {row['selected_source']}, "
            f"{row['baseline_macro_cer']:.4f} -> {row['hybrid_v3_macro_cer']:.4f}"
        )
    print(f"詳細結果: {json_path}")
    print(f"逐筆比較: {detail_path}")
    print(f"分語言摘要: {language_path}")


if __name__ == "__main__":
    main()
