import argparse
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    ignored = " ，。！？、；：,.!?;:\n\r\t()（）[]【】『』「」-_"
    result = str(text).lower()
    for character in ignored:
        result = result.replace(character, "")
    return result


def flatten_response_text(response: dict[str, Any]) -> str:
    analysis = response.get("analysis", {})
    parts = [item.get("text", "") for item in analysis.get("observations", [])]
    parts.extend(item.get("text", "") for item in analysis.get("visible_text", []))
    for item in analysis.get("context_inferences", []):
        parts.extend([item.get("description", ""), item.get("basis", "")])
    parts.extend(analysis.get("uncertainties", []))
    return " ".join(str(part) for part in parts)


def group_recall(groups: list[list[str]], output: str) -> tuple[int, int, list[list[str]]]:
    normalized_output = normalize(output)
    missing = []
    matched = 0
    for group in groups:
        if any(normalize(term) in normalized_output for term in group if term):
            matched += 1
        else:
            missing.append(group)
    return matched, len(groups), missing


def text_recall(expected: list[str], candidates: list[dict[str, Any]]) -> tuple[int, int, list[str]]:
    candidate_text = normalize(" ".join(item.get("text", "") for item in candidates))
    missing = [text for text in expected if normalize(text) not in candidate_text]
    return len(expected) - len(missing), len(expected), missing


def post_image(url: str, image_path: Path, timeout: int) -> dict[str, Any]:
    boundary = f"----TaiwanContext{uuid.uuid4().hex}"
    image_data = image_path.read_bytes()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    body = header + image_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered) + 0.5) - 1))
    return ordered[index]


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    analysis = response.get("analysis", {})
    full_text = flatten_response_text(response)
    object_output = " ".join(
        item.get("text", "") for item in analysis.get("observations", [])
    )
    context_output = " ".join(
        item.get("description", "") for item in analysis.get("context_inferences", [])
    )
    object_hit, object_total, missing_objects = group_recall(
        case.get("required_object_groups", []), object_output
    )
    context_hit, context_total, missing_context = group_recall(
        case.get("required_context_groups", []), context_output
    )
    ocr_hit, ocr_total, missing_text = text_recall(
        case.get("expected_visible_text", []), analysis.get("visible_text", [])
    )
    forbidden_hits = [
        claim for claim in case.get("forbidden_claims", [])
        if normalize(claim) in normalize(full_text)
    ]
    confirmation_correct = (
        bool(response.get("needs_confirmation"))
        == bool(case.get("expect_confirmation"))
    )
    return {
        "object_hits": object_hit,
        "object_total": object_total,
        "object_recall": object_hit / object_total if object_total else None,
        "missing_object_groups": missing_objects,
        "ocr_hits": ocr_hit,
        "ocr_total": ocr_total,
        "ocr_recall": ocr_hit / ocr_total if ocr_total else None,
        "missing_visible_text": missing_text,
        "context_hits": context_hit,
        "context_total": context_total,
        "context_recall": context_hit / context_total if context_total else None,
        "missing_context_groups": missing_context,
        "forbidden_claim_hits": forbidden_hits,
        "hallucination_free": not forbidden_hits,
        "confirmation_correct": confirmation_correct,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Taiwan Context vision API")
    parser.add_argument("--cases", default="evaluation/vision/test_cases.json")
    parser.add_argument("--output", default="evaluation/vision/results/v0.1_gpu")
    parser.add_argument("--url", default="http://127.0.0.1:8000/vision/analyze")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = suite.get("cases", [])
    results = []
    warning_counts: Counter[str] = Counter()
    suite_started = time.perf_counter()

    print(f"開始評測：{len(cases)} 張圖片")
    for index, case in enumerate(cases, 1):
        image_path = cases_path.parent / case["image_file"]
        print(f"[{index:02d}/{len(cases):02d}] {case['id']}", end=" ", flush=True)
        if not image_path.exists():
            print("MISSING_IMAGE")
            results.append({"case": case, "status": "missing_image"})
            continue
        try:
            response = post_image(args.url, image_path, args.timeout)
            checks = evaluate_case(case, response)
            warning_counts.update(response.get("warnings", []))
            results.append({
                "case": case,
                "status": "completed",
                "response": response,
                "checks": checks,
            })
            print(
                f"OK {response.get('processing_seconds', 0):.3f}s "
                f"confirm={response.get('needs_confirmation')}"
            )
        except Exception as exc:
            print(f"API_ERROR: {exc}")
            results.append({"case": case, "status": "api_error", "error": str(exc)})

    completed = [item for item in results if item["status"] == "completed"]
    latencies = [float(item["response"]["processing_seconds"]) for item in completed]
    steady = latencies[1:] if len(latencies) > 1 else []
    object_hits = sum(item["checks"]["object_hits"] for item in completed)
    object_total = sum(item["checks"]["object_total"] for item in completed)
    ocr_hits = sum(item["checks"]["ocr_hits"] for item in completed)
    ocr_total = sum(item["checks"]["ocr_total"] for item in completed)
    context_hits = sum(item["checks"]["context_hits"] for item in completed)
    context_total = sum(item["checks"]["context_total"] for item in completed)
    hallucination_free = sum(item["checks"]["hallucination_free"] for item in completed)
    confirmation_correct = sum(item["checks"]["confirmation_correct"] for item in completed)
    summary = {
        "suite": suite.get("suite"),
        "total_cases": len(cases),
        "completed_cases": len(completed),
        "missing_image_count": sum(item["status"] == "missing_image" for item in results),
        "api_error_count": sum(item["status"] == "api_error" for item in results),
        "object_recall": ratio(object_hits, object_total),
        "ocr_exact_text_recall": ratio(ocr_hits, ocr_total),
        "context_keyword_recall": ratio(context_hits, context_total),
        "hallucination_free_rate": ratio(hallucination_free, len(completed)),
        "confirmation_accuracy": ratio(confirmation_correct, len(completed)),
        "warning_counts": dict(warning_counts),
        "warmup_processing_seconds": latencies[0] if latencies else None,
        "steady_state_sample_count": len(steady),
        "steady_state_mean_processing_seconds": statistics.mean(steady) if steady else None,
        "steady_state_median_processing_seconds": statistics.median(steady) if steady else None,
        "steady_state_p95_processing_seconds": percentile_95(steady),
        "suite_wall_seconds": time.perf_counter() - suite_started,
    }
    payload = {"summary": summary, "results": results}
    (output_dir / "vision_evaluation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "human_review_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "id", "object_accuracy_1_5", "ocr_accuracy_1_5",
            "context_reasonableness_1_5", "hallucination_risk_1_5",
            "safety_presentation_1_5", "notes"
        ])
        for item in completed:
            writer.writerow([item["case"]["id"], "", "", "", "", "", ""])

    print("\n=== 視界有解 v0.1 評測完成 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"結果目錄: {output_dir}")


if __name__ == "__main__":
    main()
