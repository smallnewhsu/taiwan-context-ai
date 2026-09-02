import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效JSON") from exc
    return records


def public_sample(record):
    stable_id = hashlib.sha256(record["feedback_id"].encode("utf-8")).hexdigest()[:16]
    return {
        "sample_id": f"tc-{stable_id}",
        "domain": "family_daily_life",
        "baseline_asr": record["baseline_text"].strip(),
        "prompted_asr": record["prompt_text"].strip(),
        "reference_text": record["corrected_text"].strip(),
        "selected_source": record["selected_source"],
        "error_reasons": record.get("reasons", []),
        "consent_verified": True,
        "human_reviewed": True,
        "raw_audio_included": False,
    }


def validate(records):
    eligible = []
    skipped = Counter()
    seen_feedback = set()
    seen_content = set()
    required = {
        "feedback_id", "baseline_text", "prompt_text", "corrected_text",
        "selected_source", "consent_to_dataset", "review_status"
    }
    for record in records:
        if not required.issubset(record):
            skipped["missing_required_fields"] += 1
            continue
        if not record["consent_to_dataset"]:
            skipped["no_dataset_consent"] += 1
            continue
        if record["review_status"] != "approved":
            skipped["not_approved"] += 1
            continue
        if record["feedback_id"] in seen_feedback:
            skipped["duplicate_feedback_id"] += 1
            continue
        fingerprint = hashlib.sha256(
            re.sub(r"\s+", "", record["corrected_text"]).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_content:
            skipped["duplicate_reference_text"] += 1
            continue
        seen_feedback.add(record["feedback_id"])
        seen_content.add(fingerprint)
        eligible.append(public_sample(record))
    return eligible, skipped


def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_csv(path, samples):
    fields = [
        "sample_id", "domain", "baseline_asr", "prompted_asr",
        "reference_text", "selected_source", "error_reasons",
        "consent_verified", "human_reviewed", "raw_audio_included"
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = dict(sample)
            row["error_reasons"] = "|".join(row["error_reasons"])
            writer.writerow(row)


def dataset_card(version, count):
    return f"""# Taiwan Context Dataset — {version}

## Status

Private pilot release for pipeline validation. This release contains {count}
human-reviewed text sample(s) and is not large enough for model-performance
claims or public benchmarking.

## Contents

- Baseline ASR output
- Taiwan-context-prompted ASR output
- Human-corrected reference text
- ASR error reasons and selected candidate source

Raw audio, original filenames, feedback identifiers, and personal identifiers
are not included.

## Governance

Every exported sample has explicit dataset consent and human approval. Runtime
feedback and audit logs remain private and are not included in this release.

## Intended use

Prompt evaluation, candidate reranking research, error analysis, and future
ASR adaptation after the dataset becomes sufficiently large and diverse.

## Limitations

The current collection may be small, domain-specific, and demographically
unbalanced. Do not use this pilot release to claim broad Taiwanese Hokkien
coverage or production-level accuracy.
"""


def main():
    parser = argparse.ArgumentParser(description="Export approved Taiwan Context data")
    parser.add_argument(
        "--input",
        default="datasets/taiwan_context/feedback/approved_feedback.jsonl",
    )
    parser.add_argument(
        "--output-root", default="datasets/taiwan_context/releases"
    )
    parser.add_argument("--version", default="v0.1.0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"找不到核准資料: {source}")

    output_dir = Path(args.output_root) / args.version
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"輸出版本已存在: {output_dir}；若確定覆蓋請加 --force")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = read_jsonl(source)
    samples, skipped = validate(raw_records)
    if not samples:
        raise SystemExit("沒有符合『已同意＋已核准』條件的資料")

    write_jsonl(output_dir / "dataset.jsonl", samples)
    write_csv(output_dir / "dataset.csv", samples)
    (output_dir / "dataset_card.md").write_text(
        dataset_card(args.version, len(samples)), encoding="utf-8"
    )
    statistics = {
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_records": len(raw_records),
        "exported_samples": len(samples),
        "skipped_records": dict(skipped),
        "raw_audio_included": False,
        "all_samples_consented": True,
        "all_samples_human_reviewed": True,
        "selected_source_counts": dict(Counter(x["selected_source"] for x in samples)),
        "error_reason_counts": dict(
            Counter(reason for x in samples for reason in x["error_reasons"])
        ),
    }
    (output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== Taiwan Context Dataset匯出完成 ===")
    print(f"版本: {args.version}")
    print(f"來源資料: {len(raw_records)}")
    print(f"匯出資料: {len(samples)}")
    print(f"略過資料: {sum(skipped.values())}")
    print(f"輸出目錄: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
