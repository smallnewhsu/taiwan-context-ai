import argparse
import csv
import json
import mimetypes
import statistics
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def normalize(value: Any) -> str:
    text = str(value or "").lower()
    for character in " ，。！？、；：,.!?;:\n\r\t()（）[]【】『』「」-_～~":
        text = text.replace(character, "")
    return text


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered) + 0.5) - 1))
    return ordered[index]


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    return read_response(request, timeout)


def post_multipart(
    url: str,
    file_field: str,
    file_path: Path,
    fields: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    boundary = f"----TaiwanContext{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                "utf-8"
            )
        )
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\nContent-Type: {mime}\r\n\r\n'
        ).encode("utf-8")
    )
    chunks.append(file_path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        url,
        data=b"".join(chunks),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return read_response(request, timeout)


def read_response(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc)) from exc


def check_answer(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer_data = response.get("answer", {})
    answer = str(answer_data.get("answer", ""))
    normalized = normalize(answer)
    groups = case.get("required_answer_groups", [])
    missing_groups = [
        group
        for group in groups
        if not any(normalize(term) in normalized for term in group if term)
    ]
    forbidden_hits = [
        claim
        for claim in case.get("forbidden_claims", [])
        if normalize(claim) in normalized
    ]
    expected_uncertainty = bool(case.get("expect_uncertainty"))
    returned_uncertainty = bool(answer_data.get("needs_confirmation")) or bool(
        answer_data.get("uncertainties")
    )
    uncertainty_correct = expected_uncertainty == returned_uncertainty
    answer_nonempty = bool(answer.strip())
    keyword_recall = (
        (len(groups) - len(missing_groups)) / len(groups) if groups else 1.0
    )
    automatic_pass = (
        answer_nonempty
        and not missing_groups
        and not forbidden_hits
        and uncertainty_correct
    )
    return {
        "answer_nonempty": answer_nonempty,
        "required_group_hits": len(groups) - len(missing_groups),
        "required_group_total": len(groups),
        "keyword_recall": keyword_recall,
        "missing_required_groups": missing_groups,
        "forbidden_claim_hits": forbidden_hits,
        "hallucination_free": not forbidden_hits,
        "expected_uncertainty": expected_uncertainty,
        "returned_uncertainty": returned_uncertainty,
        "uncertainty_correct": uncertainty_correct,
        "automatic_pass": automatic_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 視界有解 image-to-text/audio follow-up pipeline"
    )
    parser.add_argument(
        "--cases", default="evaluation/vision/test_cases_multimodal_v0.2.json"
    )
    parser.add_argument("--images", default="evaluation/vision")
    parser.add_argument(
        "--output", default="evaluation/vision/results/v0.2_gemma3_4b_gpu"
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--analyze-retries", type=int, default=2)
    parser.add_argument(
        "--image-cache",
        default="",
        help="Optional previous vision_multimodal_results.json; reuse image_response by case id",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    images_dir = Path(args.images).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = suite.get("cases", [])
    base_url = args.api_url.rstrip("/")
    results: list[dict[str, Any]] = []
    warnings: Counter[str] = Counter()
    started = time.perf_counter()
    cached_images: dict[str, dict[str, Any]] = {}
    if args.image_cache:
        cache_payload = json.loads(Path(args.image_cache).read_text(encoding="utf-8"))
        cached_images = {
            item.get("case", {}).get("id"): item.get("image_response", {})
            for item in cache_payload.get("results", [])
            if item.get("status") == "completed" and item.get("image_response")
        }

    print(f"開始多模態評測：{len(cases)} 個案例")
    for index, case in enumerate(cases, 1):
        image_path = images_dir / case["image_file"]
        print(f"[{index:02d}/{len(cases):02d}] {case['id']}", end=" ", flush=True)
        if not image_path.exists():
            print("MISSING_IMAGE")
            results.append({"case": case, "status": "missing_image"})
            continue
        try:
            analyze_started = time.perf_counter()
            if case["id"] in cached_images:
                image_response = cached_images[case["id"]]
                analyze_wall_seconds = 0.0
                image_source = "cache"
            else:
                last_error: Exception | None = None
                image_response = {}
                for attempt in range(max(1, args.analyze_retries + 1)):
                    try:
                        image_response = post_multipart(
                            f"{base_url}/vision/analyze", "file", image_path, {}, args.timeout
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.analyze_retries:
                            time.sleep(0.5)
                if last_error is not None:
                    raise last_error
                analyze_wall_seconds = round(time.perf_counter() - analyze_started, 4)
                image_source = "live"

            question_started = time.perf_counter()
            if case.get("audio_file"):
                audio_path = images_dir / case["audio_file"]
                if not audio_path.exists():
                    raise RuntimeError(f"找不到追問音檔: {audio_path}")
                follow_response = post_multipart(
                    f"{base_url}/vision/follow-up",
                    "audio",
                    audio_path,
                    {
                        "image_context": json.dumps(
                            image_response, ensure_ascii=False
                        ),
                        "relationship": case.get("relationship", "未提供"),
                    },
                    args.timeout,
                )
                input_mode = "audio"
            else:
                follow_response = post_json(
                    f"{base_url}/vision/follow-up/text",
                    {
                        "question": case["question"],
                        "image_context": image_response,
                        "relationship": case.get("relationship", "未提供"),
                    },
                    args.timeout,
                )
                input_mode = "text"
            follow_wall_seconds = round(time.perf_counter() - question_started, 4)
            checks = check_answer(case, follow_response)
            warnings.update(image_response.get("warnings", []))
            results.append(
                {
                    "case": case,
                    "status": "completed",
                    "input_mode": input_mode,
                    "image_source": image_source,
                    "image_response": image_response,
                    "follow_up_response": follow_response,
                    "checks": checks,
                    "timing": {
                        "image_wall_seconds": analyze_wall_seconds,
                        "follow_up_wall_seconds": follow_wall_seconds,
                        "end_to_end_wall_seconds": round(
                            analyze_wall_seconds + follow_wall_seconds, 4
                        ),
                    },
                }
            )
            label = "PASS" if checks["automatic_pass"] else "REVIEW"
            print(
                f"{label} mode={input_mode} "
                f"e2e={analyze_wall_seconds + follow_wall_seconds:.3f}s"
            )
        except Exception as exc:
            print(f"API_ERROR: {exc}")
            results.append(
                {"case": case, "status": "api_error", "error": str(exc)}
            )
        time.sleep(max(0.0, args.delay))

    completed = [item for item in results if item["status"] == "completed"]
    latencies = [item["timing"]["end_to_end_wall_seconds"] for item in completed]
    steady = latencies[1:] if len(latencies) > 1 else []
    total_groups = sum(item["checks"]["required_group_total"] for item in completed)
    total_hits = sum(item["checks"]["required_group_hits"] for item in completed)
    summary = {
        "suite": suite.get("suite"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "completed_cases": len(completed),
        "missing_image_count": sum(
            item["status"] == "missing_image" for item in results
        ),
        "api_error_count": sum(item["status"] == "api_error" for item in results),
        "automatic_pass_count": sum(
            item["checks"]["automatic_pass"] for item in completed
        ),
        "automatic_pass_rate": (
            sum(item["checks"]["automatic_pass"] for item in completed)
            / len(completed)
            if completed
            else None
        ),
        "answer_keyword_recall": total_hits / total_groups if total_groups else None,
        "hallucination_free_rate": (
            sum(item["checks"]["hallucination_free"] for item in completed)
            / len(completed)
            if completed
            else None
        ),
        "uncertainty_accuracy": (
            sum(item["checks"]["uncertainty_correct"] for item in completed)
            / len(completed)
            if completed
            else None
        ),
        "warmup_end_to_end_seconds": latencies[0] if latencies else None,
        "steady_state_sample_count": len(steady),
        "steady_state_mean_end_to_end_seconds": (
            statistics.mean(steady) if steady else None
        ),
        "steady_state_median_end_to_end_seconds": (
            statistics.median(steady) if steady else None
        ),
        "steady_state_p95_end_to_end_seconds": percentile_95(steady),
        "warning_counts": dict(warnings),
        "image_cache_used": bool(args.image_cache),
        "suite_wall_seconds": time.perf_counter() - started,
    }
    payload = {"summary": summary, "results": results}
    (output_dir / "vision_multimodal_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "human_review_template.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "id", "input_mode", "image_grounding_1_5",
                "question_relevance_1_5", "context_reasonableness_1_5",
                "hallucination_risk_1_5", "uncertainty_handling_1_5",
                "overall_pass_y_n", "notes"
            ]
        )
        for item in completed:
            writer.writerow(
                [item["case"]["id"], item["input_mode"], "", "", "", "", "", "", ""]
            )

    print("\n=== 視界有解多模態 v0.2 評測完成 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"結果目錄: {output_dir}")


if __name__ == "__main__":
    main()
