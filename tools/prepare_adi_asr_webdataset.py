#!/usr/bin/env python3
"""Convert ADI Taiwan Tongues WebDataset shards to the official TSV layout.

This tool intentionally keeps the generated multi-GB corpus outside the Git
repository.  It reads only the official *train* shards, filters Hokkien rows to
nan-tw/zh-nan, creates a deterministic validation split, and optionally adds
the project's frozen T51--T70 evaluation recordings as test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass
class Counters:
    scanned: int = 0
    accepted: int = 0
    train: int = 0
    validated: int = 0
    rejected_language: int = 0
    rejected_missing_audio: int = 0
    rejected_missing_text: int = 0
    rejected_bad_json: int = 0
    rejected_bad_text: int = 0
    rejected_corrupt_text: int = 0
    stripped_romanization: int = 0
    source_tars: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ADI Hokkien/Hakka WebDataset for Taiwan Tongues training"
    )
    parser.add_argument(
        "--hokkien-repo",
        type=Path,
        default=Path(r"C:\taiwan-context-data\Taiwan-Tongues-ASR-CE-dataset-hokkien"),
    )
    parser.add_argument(
        "--hakka-repo",
        type=Path,
        default=Path(r"C:\taiwan-context-data\Taiwan-Tongues-ASR-CE-dataset-hakka"),
    )
    parser.add_argument(
        "--frozen-test-source",
        type=Path,
        default=Path(r"C:\taiwan-context-ai\evaluation\asr_multilingual"),
        help="Folder containing T51-T70 audio and *_reference.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\taiwan-context-training\sample_corpus"),
    )
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=0,
        help="For a pilot run only; 0 processes all accepted samples",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Scan and report without writing audio or TSV files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of same-named generated files",
    )
    return parser.parse_args()


TAIGI_ROMANIZATION = re.compile(
    r"\s*(?:（[^）]*[A-Za-z?][^）]*）|\([^)]*[A-Za-z?][^)]*\))"
)
TAIGI_UNCLOSED_ROMANIZATION = re.compile(
    r"\s*(?:（|\()[^（）()]*[A-Za-z?][^（）()]*$"
)


def normalize_text(raw: bytes, language: str) -> tuple[str, bool]:
    text = raw.decode("utf-8-sig", errors="strict")
    text = " ".join(text.split()).strip()
    stripped = False
    if language == "taigi":
        cleaned = TAIGI_ROMANIZATION.sub("", text).strip()
        # Some source rows contain a truncated romanization annotation without
        # a closing parenthesis, for example "國之將亡（Kok tsi...".  The
        # annotation is not an ASR target and can be removed safely.
        cleaned = TAIGI_UNCLOSED_ROMANIZATION.sub("", cleaned).strip()
        stripped = cleaned != text
        text = cleaned
    return text, stripped


def sample_group(metadata: dict, key: str, language: str) -> str:
    if language == "taigi":
        return str(
            metadata.get("client_id")
            or metadata.get("speaker_id")
            or metadata.get("sentence_id")
            or key
        )
    return str(metadata.get("speaker_id") or metadata.get("voice") or key)


def choose_split(group: str, validation_percent: int) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "validated" if bucket < validation_percent else "train"


def hokkien_allowed(metadata: dict) -> bool:
    locale = str(metadata.get("locale") or "").strip().lower()
    audio_path = str(
        metadata.get("path")
        or metadata.get("audio_relpath")
        or metadata.get("audio_rel")
        or ""
    ).lower()
    if locale in {"nan-tw", "zh-nan", "nan"}:
        return True
    if "nan-tw" in audio_path or "zh-nan" in audio_path:
        return True
    return False


def iter_tar_samples(tar_path: Path) -> Iterator[tuple[str, dict[str, bytes]]]:
    current_key: str | None = None
    current: dict[str, bytes] = {}
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            suffix = member_path.suffix.lower()
            key = member_path.name[: -len(suffix)] if suffix else member_path.name
            if current_key is not None and key != current_key:
                yield current_key, current
                current = {}
            current_key = key
            extracted: BinaryIO | None = archive.extractfile(member)
            if extracted is not None:
                current[suffix] = extracted.read()
    if current_key is not None:
        yield current_key, current


def prepare_language(
    language: str,
    repo: Path,
    output_root: Path,
    validation_percent: int,
    limit: int,
    audit_only: bool,
    overwrite: bool,
) -> tuple[Counters, dict[str, list[tuple[str, str]]]]:
    dataset_name = "train_ds_taigi" if language == "taigi" else "train_ds_hakka"
    dataset_dir = output_root / dataset_name
    clips_dir = dataset_dir / "clips"
    if not repo.is_dir():
        raise FileNotFoundError(f"Dataset repository not found: {repo}")
    tar_paths = sorted((repo / "train").glob("train-*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No train/train-*.tar found under {repo}")
    if not audit_only:
        clips_dir.mkdir(parents=True, exist_ok=True)

    counters = Counters(source_tars=[str(path) for path in tar_paths])
    rows: dict[str, list[tuple[str, str]]] = {"train": [], "validated": [], "test": []}

    stop = False
    for tar_path in tar_paths:
        for key, parts in iter_tar_samples(tar_path):
            counters.scanned += 1
            audio_suffixes = sorted(set(parts).intersection(AUDIO_SUFFIXES))
            if len(audio_suffixes) != 1:
                counters.rejected_missing_audio += 1
                continue
            if ".txt" not in parts:
                counters.rejected_missing_text += 1
                continue
            try:
                metadata = json.loads(parts.get(".json", b"{}").decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counters.rejected_bad_json += 1
                continue
            if language == "taigi" and not hokkien_allowed(metadata):
                counters.rejected_language += 1
                continue
            try:
                sentence, stripped = normalize_text(parts[".txt"], language)
            except UnicodeDecodeError:
                counters.rejected_bad_text += 1
                continue
            if not sentence or "\ufffd" in sentence:
                counters.rejected_bad_text += 1
                continue
            # ASCII '?' in the ADI Hokkien text is a replacement character
            # produced by damaged source orthography, not Chinese punctuation.
            # Its original Han character cannot be reconstructed reliably, so
            # exclude the row instead of teaching the corruption to the model.
            if language == "taigi" and "?" in sentence:
                counters.rejected_corrupt_text += 1
                continue
            if stripped:
                counters.stripped_romanization += 1

            split = choose_split(
                sample_group(metadata, key, language), validation_percent
            )
            audio_suffix = audio_suffixes[0]
            filename = f"adi_{language}_{key}{audio_suffix}"
            destination = clips_dir / filename
            if not audit_only:
                if destination.exists() and not overwrite:
                    raise FileExistsError(
                        f"Output already exists: {destination}; use --overwrite or a fresh output folder"
                    )
                destination.write_bytes(parts[audio_suffix])
            rows[split].append((filename, sentence))
            counters.accepted += 1
            setattr(counters, split, getattr(counters, split) + 1)
            if limit and counters.accepted >= limit:
                stop = True
                break
        if stop:
            break
    return counters, rows


def find_frozen_audio(folder: Path, case_id: str) -> Path:
    matches = [
        path
        for suffix in AUDIO_SUFFIXES
        if (path := folder / f"{case_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one frozen audio for {case_id}; found {len(matches)}"
        )
    return matches[0]


def frozen_reference(folder: Path, case_id: str) -> str:
    for name in (
        f"{case_id}_reference.txt",
        f"{case_id}_ground_truth.txt",
        f"{case_id}_transcript.txt",
    ):
        path = folder / name
        if path.is_file():
            return " ".join(path.read_text(encoding="utf-8-sig").split())
    raise FileNotFoundError(f"Frozen reference not found for {case_id}")


def add_frozen_test(
    language: str,
    folder: Path,
    output_root: Path,
    rows: dict[str, list[tuple[str, str]]],
    overwrite: bool,
) -> int:
    dataset_name = "train_ds_taigi" if language == "taigi" else "train_ds_hakka"
    clips_dir = output_root / dataset_name / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    numbers = range(51, 61) if language == "taigi" else range(61, 71)
    for number in numbers:
        case_id = f"T{number}"
        source = find_frozen_audio(folder, case_id)
        destination = clips_dir / f"frozen_{source.name}"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}; use --overwrite or a fresh output folder"
            )
        shutil.copy2(source, destination)
        rows["test"].append((destination.name, frozen_reference(folder, case_id)))
    return len(list(numbers))


def write_tsvs(
    language: str, output_root: Path, rows: dict[str, list[tuple[str, str]]]
) -> None:
    dataset_name = "train_ds_taigi" if language == "taigi" else "train_ds_hakka"
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validated", "test"):
        with (dataset_dir / f"{split}.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["path", "sentence"])
            writer.writerows(sorted(rows[split]))


def main() -> None:
    args = parse_args()
    if not 1 <= args.validation_percent <= 30:
        raise ValueError("--validation-percent must be between 1 and 30")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (
        args.overwrite or args.audit_only
    ):
        raise FileExistsError(
            f"Output folder is not empty: {args.output_dir}; use a fresh folder or --overwrite"
        )

    report: dict[str, object] = {
        "source": "adi-gov-tw Taiwan Tongues ASR CE public datasets",
        "validation_percent": args.validation_percent,
        "validation_assignment": "deterministic SHA-256 group hash",
        "frozen_test_policy": "T51-T70 are test-only and never train/validated",
        "decoder_prefix": "zh (public v1.0 has no nan/hak token)",
        "languages": {},
    }

    for language, repo in (("taigi", args.hokkien_repo), ("hakka", args.hakka_repo)):
        counters, rows = prepare_language(
            language=language,
            repo=repo,
            output_root=args.output_dir,
            validation_percent=args.validation_percent,
            limit=args.limit_per_language,
            audit_only=args.audit_only,
            overwrite=args.overwrite,
        )
        frozen_count = 0
        if not args.audit_only:
            frozen_count = add_frozen_test(
                language,
                args.frozen_test_source,
                args.output_dir,
                rows,
                args.overwrite,
            )
            write_tsvs(language, args.output_dir, rows)
        language_report = vars(counters)
        language_report["frozen_test"] = frozen_count
        if language == "hakka":
            language_report["limitation"] = (
                "Official Hakka metadata has no stable speaker identifier; "
                "validation is sample-hash split and cannot prove speaker isolation."
            )
        report["languages"][language] = language_report

    if not args.audit_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "corpus_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print("=== ADI ASR corpus audit/preparation complete ===")
    for language, values in report["languages"].items():
        print(
            f"{language}: scanned={values['scanned']}, accepted={values['accepted']}, "
            f"train={values['train']}, validated={values['validated']}, "
            f"rejected_language={values['rejected_language']}, "
            f"rejected_corrupt_text={values['rejected_corrupt_text']}"
        )
    print("Mode:", "audit only" if args.audit_only else "corpus written")
    if not args.audit_only:
        print("Output:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
