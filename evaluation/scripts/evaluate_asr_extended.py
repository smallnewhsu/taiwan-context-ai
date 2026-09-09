"""T22-T50 Taiwan Tongues ASR CE extended GPU evaluation.

Run from the repository root:
    python evaluation/scripts/evaluate_asr_extended.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
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


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_text(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def validate_dataset(folder: Path) -> None:
    expected = set(LANGUAGE_BY_ID)
    audio_by_id: dict[str, list[Path]] = defaultdict(list)
    for extension in (".wav", ".mp3", ".flac", ".m4a", ".aac"):
        for path in folder.glob(f"*{extension}"):
            audio_by_id[path.stem.upper()].append(path)

    missing_audio = sorted(expected - set(audio_by_id))
    duplicate_audio = sorted(key for key, paths in audio_by_id.items() if key in expected and len(paths) != 1)
    unexpected_audio = sorted(set(audio_by_id) - expected)
    missing_reference = sorted(
        case_id for case_id in expected if not (folder / f"{case_id}_reference.txt").is_file()
    )
    empty_reference = sorted(
        case_id for case_id in expected
        if (folder / f"{case_id}_reference.txt").is_file()
        and not (folder / f"{case_id}_reference.txt").read_text(encoding="utf-8-sig").strip()
    )
    problems = []
    if missing_audio:
        problems.append(f"缺少音檔: {', '.join(missing_audio)}")
    if duplicate_audio:
        problems.append(f"同編號有多個音檔: {', '.join(duplicate_audio)}")
    if unexpected_audio:
        problems.append(f"資料夾含非 T22-T50 音檔: {', '.join(unexpected_audio)}")
    if missing_reference:
        problems.append(f"缺少標準逐字稿: {', '.join(missing_reference)}")
    if empty_reference:
        problems.append(f"空白標準逐字稿: {', '.join(empty_reference)}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("資料檢查通過：29 個音檔與 29 個標準逐字稿。")


def run_transcription(repo_root: Path, folder: Path, result_prefix: str) -> tuple[Path, Path | None]:
    asr_dir = repo_root / "services" / "asr"
    evaluator = asr_dir / "evaluate_asr.py"
    if not evaluator.is_file():
        raise SystemExit(f"找不到 ASR 評測程式: {evaluator}")
    help_result = subprocess.run(
        [sys.executable, str(evaluator), "--help"],
        cwd=asr_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = f"{help_result.stdout}\n{help_result.stderr}"
    result_json = asr_dir / f"{result_prefix}_results.json"
    latency_csv: Path | None = asr_dir / f"{result_prefix}_latency.csv"
    if "--result-prefix" in help_text:
        command = [
            sys.executable, str(evaluator), str(folder),
            "--result-prefix", result_prefix,
        ]
    elif "--output" in help_text:
        # 新版 evaluate_asr.py：--output 直接指定 JSON 檔案。
        command = [
            sys.executable, str(evaluator), str(folder),
            "--output", str(result_json),
        ]
        latency_csv = None
    else:
        raise SystemExit(
            "無法辨識 services/asr/evaluate_asr.py 的命令列介面；"
            "預期支援 --output 或 --result-prefix。"
        )
    print("啟動 Taiwan Tongues ASR CE GPU 評測…")
    subprocess.run(command, cwd=asr_dir, check=True)
    if not result_json.is_file():
        # 部分新版程式雖接受 --output，仍固定使用此檔名。
        fixed_result = asr_dir / "asr_comparison_results.json"
        if fixed_result.is_file():
            result_json = fixed_result
        else:
            raise SystemExit(
                "辨識完成，但找不到結果檔；已檢查: "
                f"{result_json}、{fixed_result}"
            )
    if latency_csv is not None and not latency_csv.is_file():
        latency_csv = None
    return result_json, latency_csv


def aggregate(result_json: Path) -> dict:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    results = payload.get("detailed_results") or payload.get("results") or []
    grouped: dict[str, list[dict]] = defaultdict(list)
    unknown_ids = []
    for item in results:
        audio_name = item.get("audio_file") or item.get("file_name") or item.get("filename") or ""
        case_id = Path(audio_name).stem.upper()
        language = LANGUAGE_BY_ID.get(case_id)
        if language is None:
            unknown_ids.append(case_id)
            continue
        item["case_id"] = case_id
        item["language_type"] = language
        grouped[language].append(item)

    language_summary = {}
    for language, items in grouped.items():
        valid = [item for item in items if item.get("cer_result")]
        latencies = [item.get("latency", {}) for item in items if item.get("latency")]
        cer_values = [
            float(item["cer_result"].get("cer_rate", item["cer_result"].get("cer", 0)))
            for item in valid
        ]
        total_errors = sum(
            int(item["cer_result"].get("total_errors", 0)) for item in valid
        )
        total_chars = sum(
            int(item["cer_result"].get("total_chars", item["cer_result"].get("reference_length", 0)))
            for item in valid
        )
        inference = [
            float(item.get("inference_seconds", 0)) for item in latencies
            if item.get("inference_seconds") is not None
        ]
        rtfs = [float(item.get("rtf", 0)) for item in latencies if item.get("rtf") is not None]
        language_summary[language] = {
            "sample_count": len(items),
            "cer_sample_count": len(valid),
            "macro_average_cer": mean(cer_values),
            "micro_cer": total_errors / total_chars if total_chars else None,
            "total_errors": total_errors,
            "total_reference_characters": total_chars,
            "average_inference_seconds": mean(inference),
            "p95_inference_seconds": percentile(inference, 95),
            "average_rtf": mean(rtfs),
        }

    payload["suite"] = "Taiwan Context AI ASR Extended T22-T50"
    payload["language_summary"] = language_summary
    payload["evaluation_scope"] = {
        "expected_case_count": 29,
        "case_range": "T22-T50",
        "language_mapping_version": "v1",
        "unexpected_case_ids": unknown_ids,
    }
    payload["detailed_results"] = sorted(results, key=lambda item: item.get("case_id", ""))
    return payload


def write_latency_csv(payload: dict, output_path: Path) -> None:
    fields = [
        "case_id", "language_type", "audio_duration_seconds",
        "inference_seconds", "rtf", "is_warmup_sample", "device", "compute_type",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in payload.get("detailed_results", []):
            latency = item.get("latency") or {}
            writer.writerow({
                "case_id": item.get("case_id", ""),
                "language_type": item.get("language_type", ""),
                **{field: latency.get(field, "") for field in fields[2:]},
            })


def write_language_csv(payload: dict, output_path: Path) -> None:
    fields = [
        "language_type", "sample_count", "cer_sample_count", "macro_average_cer",
        "micro_cer", "total_errors", "total_reference_characters",
        "average_inference_seconds", "p95_inference_seconds", "average_rtf",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for language, values in payload["language_summary"].items():
            writer.writerow({"language_type": language, **values})


def main() -> None:
    parser = argparse.ArgumentParser(description="T22-T50 擴充 ASR GPU 評測")
    parser.add_argument("--folder", default="evaluation/asr_extended")
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--result-prefix", default="asr_extended_gpu_29")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    folder = (repo_root / args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"資料夾不存在: {folder}")
    validate_dataset(folder)

    source_json = repo_root / "services" / "asr" / f"{args.result_prefix}_results.json"
    fixed_source_json = repo_root / "services" / "asr" / "asr_comparison_results.json"
    source_latency: Path | None = repo_root / "services" / "asr" / f"{args.result_prefix}_latency.csv"
    if not args.skip_transcription:
        source_json, source_latency = run_transcription(repo_root, folder, args.result_prefix)
    elif not source_json.is_file():
        if fixed_source_json.is_file():
            source_json = fixed_source_json
        else:
            raise SystemExit(
                "--skip-transcription 找不到先前結果；已檢查: "
                f"{source_json}、{fixed_source_json}"
            )

    payload = aggregate(source_json)
    output_json = folder / "asr_extended_results_gpu_29.json"
    output_latency = folder / "asr_extended_latency_gpu_29.csv"
    output_language = folder / "asr_extended_language_summary_gpu_29.csv"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if source_latency is not None and source_latency.is_file():
        shutil.copyfile(source_latency, output_latency)
    else:
        write_latency_csv(payload, output_latency)
    write_language_csv(payload, output_language)

    summary = payload.get("summary", {})
    print("\n=== T22-T50 擴充評測完成 ===")
    print(f"總案例數: {summary.get('total_files', 0)}")
    print(f"成功 CER 比對: {summary.get('files_with_cer', 0)}")
    print(f"整體平均 CER: {summary.get('average_cer', 0):.4f}")
    after = summary.get("latency_after_warmup", {})
    print(f"排除暖機後平均推論時間: {after.get('average_inference_seconds', 0):.4f} 秒")
    print(f"排除暖機後 P95: {after.get('p95_inference_seconds', 0):.4f} 秒")
    print("\n各語言 CER:")
    for language, values in payload["language_summary"].items():
        macro = values["macro_average_cer"]
        micro = values["micro_cer"]
        print(
            f"- {language}: n={values['sample_count']}, "
            f"macro={metric_text(macro)}, micro={metric_text(micro)}"
        )
    print(f"\n詳細結果: {output_json}")
    print(f"延遲結果: {output_latency}")
    print(f"分語言摘要: {output_language}")


if __name__ == "__main__":
    main()
