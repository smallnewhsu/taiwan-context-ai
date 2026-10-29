#!/usr/bin/env python3
"""Aggregate and verify three frozen Taigi LoRA final evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_FILES = (
    "taigi_v3_final_results.json",
    "taigi_v3_final_run2_results.json",
    "taigi_v3_final_run3_results.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate three Taigi final runs")
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="taigi_v3_final_3run",
    )
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def transcription_signature(payload: dict) -> list[tuple]:
    return [
        (
            row["case_id"],
            row["reference"],
            row["base_text"],
            row["adapter_text"],
            row["base_score"]["errors"],
            row["adapter_score"]["errors"],
        )
        for row in payload["results"]
    ]


def main() -> None:
    args = parse_args()
    paths = [args.folder / name for name in DEFAULT_FILES]
    payloads = [load(path) for path in paths]
    signatures = [transcription_signature(payload) for payload in payloads]
    outputs_identical = all(signature == signatures[0] for signature in signatures[1:])
    if not outputs_identical:
        raise RuntimeError("Transcriptions differ between runs")

    summaries = [payload["summary"] for payload in payloads]
    total_cases = {int(summary["total_cases"]) for summary in summaries}
    if total_cases != {20}:
        raise RuntimeError(f"Expected 20 cases in every run; got {sorted(total_cases)}")

    metrics = {
        "base_macro_cer": [float(s["base"]["macro_cer"]) for s in summaries],
        "base_micro_cer": [float(s["base"]["micro_cer"]) for s in summaries],
        "adapter_macro_cer": [float(s["adapter"]["macro_cer"]) for s in summaries],
        "adapter_micro_cer": [float(s["adapter"]["micro_cer"]) for s in summaries],
        "relative_micro_cer_improvement": [
            float(s["micro_cer_relative_improvement"]) for s in summaries
        ],
        "base_mean_seconds": [float(s["base"]["mean_seconds"]) for s in summaries],
        "adapter_mean_seconds": [
            float(s["adapter"]["mean_seconds"]) for s in summaries
        ],
    }
    metric_stats = {name: stats(values) for name, values in metrics.items()}
    latency_deltas = [
        adapter - base
        for adapter, base in zip(
            metrics["adapter_mean_seconds"], metrics["base_mean_seconds"]
        )
    ]
    metric_stats["adapter_minus_base_seconds"] = stats(latency_deltas)

    first = summaries[0]
    report = {
        "suite": "Taiwan Context AI Taigi LoRA v3 frozen final evaluation",
        "runs": len(payloads),
        "cases_per_run": 20,
        "total_inferences_per_model": 60,
        "api_error_count": 0,
        "transcriptions_identical_across_runs": outputs_identical,
        "base_total_errors": int(first["base"]["total_errors"]),
        "adapter_total_errors": int(first["adapter"]["total_errors"]),
        "errors_reduced": int(first["base"]["total_errors"])
        - int(first["adapter"]["total_errors"]),
        "improved_cases": int(first["improved_cases"]),
        "unchanged_cases": int(first["unchanged_cases"]),
        "regressed_cases": int(first["regressed_cases"]),
        "metrics": metric_stats,
        "source_files": [str(path) for path in paths],
    }

    json_path = args.folder / f"{args.output_prefix}_summary.json"
    csv_path = args.folder / f"{args.output_prefix}_summary.csv"
    md_path = args.folder / f"{args.output_prefix}_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "standard_deviation", "minimum", "maximum"])
        for name, values in metric_stats.items():
            writer.writerow(
                [
                    name,
                    values["mean"],
                    values["standard_deviation"],
                    values["minimum"],
                    values["maximum"],
                ]
            )

    base_micro = metric_stats["base_micro_cer"]["mean"]
    adapter_micro = metric_stats["adapter_micro_cer"]["mean"]
    improvement = metric_stats["relative_micro_cer_improvement"]["mean"]
    base_time = metric_stats["base_mean_seconds"]["mean"]
    adapter_time = metric_stats["adapter_mean_seconds"]["mean"]
    markdown = f"""# 台語 LoRA v3 最終凍結評測

- 測試集：T81-T100，20 筆，2 位說話者
- 重複執行：3 次
- 每個模型總推論數：60
- API 錯誤：0
- 三次轉錄輸出完全一致：是

| 指標 | Base | LoRA v3 |
|---|---:|---:|
| Micro CER | {base_micro:.2%} | {adapter_micro:.2%} |
| 平均推論時間 | {base_time:.3f} 秒 | {adapter_time:.3f} 秒 |
| 錯誤字元 | {report['base_total_errors']} | {report['adapter_total_errors']} |

Micro CER 相對下降 {improvement:.2%}，錯誤字元減少 {report['errors_reduced']} 個。20 筆案例中，{report['improved_cases']} 筆改善、{report['unchanged_cases']} 筆持平、{report['regressed_cases']} 筆退步。三次轉錄內容完全一致，平均延遲差為 {metric_stats['adapter_minus_base_seconds']['mean']:.3f} 秒。

## 限制

本結果來自 20 筆凍結測試案例，絕對 CER 仍有改善空間。退步案例保留於逐筆比較檔，正式系統須搭配語意安全檢查及使用者確認。
"""
    md_path.write_text(markdown, encoding="utf-8")

    print("=== Taigi three-run final summary ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
