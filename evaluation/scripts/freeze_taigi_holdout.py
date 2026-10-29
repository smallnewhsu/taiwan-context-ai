#!/usr/bin/env python3
"""Validate and freeze the T81-T100 Taiwanese final holdout set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import librosa


AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")
EXPECTED_IDS = [f"T{number}" for number in range(81, 101)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze T81-T100 final holdout")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\taigi_final_holdout_v1.csv"
        ),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\audio"
        ),
    )
    parser.add_argument(
        "--training-tsv",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\sample_corpus\train_ds_taigi\train.tsv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            r"C:\taiwan-context-ai\evaluation\asr_finetune\final_holdout_v1\frozen_manifest.json"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_spec(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Spec not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle))
    header_index = next(
        (
            index
            for index, row in enumerate(raw_rows)
            if "case_id" in row and "reference_text" in row
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Could not find a row containing case_id and reference_text")
    header = raw_rows[header_index]
    required = {
        "case_id",
        "reference_text",
        "literal_meaning",
        "speaker_id",
        "recording_status",
    }
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    rows = [
        dict(zip(header, row))
        for row in raw_rows[header_index + 1 :]
        if row and row[0].strip()
    ]
    return header, rows


def read_training_sentences(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training TSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["sentence"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("sentence", "").strip()
        }


def find_audio(folder: Path, case_id: str) -> Path:
    matches = [folder / f"{case_id}{suffix}" for suffix in AUDIO_SUFFIXES]
    matches = [path for path in matches if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"{case_id}: expected exactly one audio file, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    _, rows = read_spec(args.spec)
    ids = [row["case_id"].strip() for row in rows]
    if ids != EXPECTED_IDS:
        raise ValueError(f"Expected case IDs {EXPECTED_IDS}; received {ids}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case_id found")

    training_sentences = read_training_sentences(args.training_tsv)
    frozen_cases: list[dict[str, object]] = []
    for row in rows:
        case_id = row["case_id"].strip()
        reference = row["reference_text"].strip()
        literal_meaning = row["literal_meaning"].strip()
        speaker_id = row["speaker_id"].strip()
        status = row["recording_status"].strip().lower()
        if not reference or not literal_meaning:
            raise ValueError(f"{case_id}: missing reference_text or literal_meaning")
        if reference in training_sentences:
            raise ValueError(f"{case_id}: exact transcript exists in training corpus")
        if not speaker_id:
            raise ValueError(f"{case_id}: speaker_id is required before freezing")
        if status not in {"reviewed", "frozen"}:
            raise ValueError(
                f"{case_id}: recording_status must be reviewed or frozen; got {status!r}"
            )
        audio_path = find_audio(args.audio_dir, case_id)
        duration = float(librosa.get_duration(path=audio_path))
        if duration <= 0.25:
            raise ValueError(f"{case_id}: audio is too short ({duration:.3f}s)")
        reference_path = args.audio_dir / f"{case_id}_reference.txt"
        reference_path.write_text(reference + "\n", encoding="utf-8")
        frozen_cases.append(
            {
                "case_id": case_id,
                "category": row.get("category", "").strip(),
                "speaker_id": speaker_id,
                "reference_text": reference,
                "literal_meaning": literal_meaning,
                "audio_file": audio_path.name,
                "audio_duration_seconds": round(duration, 4),
                "audio_sha256": sha256(audio_path),
                "reference_file": reference_path.name,
                "reference_sha256": sha256(reference_path),
            }
        )

    speaker_ids = sorted({str(case["speaker_id"]) for case in frozen_cases})
    manifest = {
        "suite": "Taiwan Context AI Taigi Final Holdout v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "spec_file": str(args.spec),
        "spec_sha256": sha256(args.spec),
        "training_tsv": str(args.training_tsv),
        "training_tsv_sha256": sha256(args.training_tsv),
        "total_cases": len(frozen_cases),
        "speaker_count": len(speaker_ids),
        "speaker_ids": speaker_ids,
        "exact_training_overlap_count": 0,
        "cases": frozen_cases,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Taigi final holdout frozen ===")
    print(f"Cases: {len(frozen_cases)}")
    print(f"Speakers: {len(speaker_ids)} ({', '.join(speaker_ids)})")
    print("Exact training overlap: 0")
    print(f"Manifest: {args.manifest}")
    print(f"Test directory: {args.audio_dir}")


if __name__ == "__main__":
    main()
