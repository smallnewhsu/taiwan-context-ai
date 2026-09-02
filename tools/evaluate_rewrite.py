#!/usr/bin/env python3
"""Evaluate the frozen Expression Rewrite v0.1 API without tuning it."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEGATION_MARKERS = ("不", "沒", "無", "未", "毋", "莫", "勿")


def load_cases(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return data, cases


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extracted_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def evaluate_constraints(case: dict[str, Any], output_text: str) -> dict[str, Any]:
    missing_all = [term for term in case.get("required_all", []) if term not in output_text]
    missing_groups = [
        group
        for group in case.get("required_any_groups", [])
        if not any(term in output_text for term in group)
    ]
    forbidden_found = [term for term in case.get("forbidden_terms", []) if term in output_text]

    source_numbers = extracted_numbers(case["original_text"])
    output_numbers = extracted_numbers(output_text)
    numbers_preserved = Counter(source_numbers) == Counter(output_numbers)

    source_has_negation = any(marker in case["original_text"] for marker in NEGATION_MARKERS)
    output_has_negation = any(marker in output_text for marker in NEGATION_MARKERS)
    negation_preserved = not source_has_negation or output_has_negation

    failures: list[str] = []
    if missing_all:
        failures.append("missing_required_terms")
    if missing_groups:
        failures.append("missing_required_concepts")
    if forbidden_found:
        failures.append("forbidden_information_added")
    if not numbers_preserved:
        failures.append("numbers_changed_or_missing")
    if not negation_preserved:
        failures.append("negation_changed_or_missing")

    return {
        "automatic_pass": not failures,
        "failures": failures,
        "missing_required_terms": missing_all,
        "missing_required_groups": missing_groups,
        "forbidden_terms_found": forbidden_found,
        "source_numbers": source_numbers,
        "output_numbers": output_numbers,
        "numbers_preserved": numbers_preserved,
        "negation_preserved": negation_preserved,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize(results: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    completed = [row for row in results if row.get("response")]
    passed = [row for row in completed if row["checks"]["automatic_pass"]]
    fallback = [row for row in completed if row["response"].get("fallback_used")]
    confirmed = [row for row in completed if row["response"].get("needs_confirmation")]
    latencies = [
        float(row["response"]["processing_seconds"])
        for row in completed
        if row["response"].get("processing_seconds") is not None
    ]
    warning_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    for row in completed:
        warning_counts.update(row["response"].get("warnings", []))
        failure_counts.update(row["checks"].get("failures", []))

    count = len(results)
    completed_count = len(completed)
    return {
        "total_cases": count,
        "completed_cases": completed_count,
        "api_error_count": count - completed_count,
        "automatic_pass_count": len(passed),
        "automatic_pass_rate": len(passed) / count if count else 0,
        "llm_candidate_accepted_count": completed_count - len(fallback),
        "llm_candidate_accepted_rate": (completed_count - len(fallback)) / completed_count if completed_count else 0,
        "safe_fallback_count": len(fallback),
        "safe_fallback_rate": len(fallback) / completed_count if completed_count else 0,
        "needs_confirmation_count": len(confirmed),
        "warning_counts": dict(warning_counts),
        "automatic_failure_counts": dict(failure_counts),
        "mean_processing_seconds": statistics.fmean(latencies) if latencies else None,
        "median_processing_seconds": statistics.median(latencies) if latencies else None,
        "p95_processing_seconds": percentile(latencies, 0.95),
        "suite_wall_seconds": elapsed,
    }


def write_human_review(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "id", "category", "original_text", "rewritten_text", "audience", "tone",
        "fallback_used", "automatic_pass", "fidelity_score_1_5",
        "audience_fit_score_1_5", "naturalness_score_1_5", "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            case = row["case"]
            response = row.get("response") or {}
            checks = row.get("checks") or {}
            writer.writerow({
                "id": case["id"],
                "category": case.get("category", ""),
                "original_text": case["original_text"],
                "rewritten_text": response.get("rewritten_text", ""),
                "audience": case["audience"],
                "tone": case["tone"],
                "fallback_used": response.get("fallback_used", ""),
                "automatic_pass": checks.get("automatic_pass", ""),
                "fidelity_score_1_5": "",
                "audience_fit_score_1_5": "",
                "naturalness_score_1_5": "",
                "reviewer_notes": "",
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 語你傳心 rewrite API v0.1")
    parser.add_argument("--cases", default="evaluation/rewrite/test_cases.json")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/expression/rewrite")
    parser.add_argument("--output-dir", default="evaluation/rewrite/results/v0.1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    suite, cases = load_cases(Path(args.cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    print(f"開始評測：{len(cases)} 個案例")
    for index, case in enumerate(cases, start=1):
        payload = {key: case[key] for key in ("original_text", "audience", "tone", "scenario")}
        row: dict[str, Any] = {"case": case, "request": payload}
        try:
            response = post_json(args.api_url, payload, args.timeout)
            output_text = str(response.get("rewritten_text", ""))
            row["response"] = response
            row["checks"] = evaluate_constraints(case, output_text)
            status = "PASS" if row["checks"]["automatic_pass"] else "FAIL"
            mode = "fallback" if response.get("fallback_used") else "LLM"
            print(f"[{index:02d}/{len(cases):02d}] {case['id']} {status} ({mode})")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            row["error"] = f"HTTPError {exc.code}: {detail}"
            print(f"[{index:02d}/{len(cases):02d}] {case['id']} API_ERROR: {row['error']}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index:02d}/{len(cases):02d}] {case['id']} API_ERROR: {exc}")
        results.append(row)
        if index < len(cases) and args.delay:
            time.sleep(args.delay)

    elapsed = time.perf_counter() - started
    summary = summarize(results, elapsed)
    report = {
        "suite": suite.get("suite"),
        "description": suite.get("description"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "summary": summary,
        "results": results,
    }
    (output_dir / "rewrite_evaluation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_human_review(output_dir / "human_review_template.csv", results)

    print("\n=== 語你傳心 v0.1 評測完成 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"結果目錄: {output_dir.resolve()}")
    return 1 if summary["api_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
