import argparse
import glob
import json
import os
import time

import librosa
from faster_whisper import WhisperModel

from cer import compare_texts
from evaluate_asr import (
    chinese_number_to_arabic,
    find_original_transcript,
    remove_special_characters_by_dataset_name,
    replace_words,
    s2tw,
)


AUDIO_PATTERNS = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac")


def normalize(text):
    return remove_special_characters_by_dataset_name(
        chinese_number_to_arabic(s2tw.convert(replace_words(text)))
    ).lower()


def transcribe(model, audio, prompt=""):
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
    segments = list(segments)
    elapsed = time.perf_counter() - started
    text = normalize("".join(segment.text for segment in segments))

    weights = [max(segment.end - segment.start, 0.001) for segment in segments]
    total_weight = sum(weights)
    confidence = (
        sum(segment.avg_logprob * weight for segment, weight in zip(segments, weights))
        / total_weight
        if total_weight
        else -999.0
    )
    no_speech = (
        sum(segment.no_speech_prob * weight for segment, weight in zip(segments, weights))
        / total_weight
        if total_weight
        else 1.0
    )
    return {
        "text": text,
        "confidence": confidence,
        "no_speech_probability": no_speech,
        "inference_seconds": elapsed,
    }


def cer_payload(reference, hypothesis):
    result = compare_texts(reference, hypothesis)
    if not result:
        return None
    return {
        "cer": result.cer_rate,
        "correct_rate": result.correct_rate,
        "errors": result.total_errors,
        "substitutions": result.substitutions_count,
        "deletions": result.deletions_count,
        "insertions": result.insertions_count,
        "characters": result.total_chars,
    }


def collect_audio(folder):
    files = []
    for pattern in AUDIO_PATTERNS:
        files.extend(glob.glob(os.path.join(folder, pattern)))
        files.extend(glob.glob(os.path.join(folder, pattern.upper())))
    return sorted(set(files), key=lambda path: os.path.basename(path).lower())


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="比較無提示與臺語提示ASR，並以模型信心分數進行路由"
    )
    parser.add_argument("folder", help="包含音檔與參考逐字稿的資料夾")
    parser.add_argument("--prompt-file", required=True, help="UTF-8提示詞檔")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="提示版信心分數至少高出此值才採用，預設0.0",
    )
    parser.add_argument(
        "--output",
        default="router_results.json",
        help="JSON輸出路徑",
    )
    args = parser.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8-sig") as handle:
        prompt = handle.read().strip()

    audio_files = collect_audio(args.folder)
    if not audio_files:
        raise SystemExit(f"找不到音檔: {args.folder}")

    load_started = time.perf_counter()
    model = WhisperModel("models", device="cuda", compute_type="float16")
    model_load_seconds = time.perf_counter() - load_started

    details = []
    for index, audio_file in enumerate(audio_files, 1):
        name = os.path.basename(audio_file)
        print(f"處理 {index}/{len(audio_files)}: {name}")
        audio, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
        duration = len(audio) / float(sample_rate)

        baseline = transcribe(model, audio)
        prompted = transcribe(model, audio, prompt)
        use_prompt = prompted["confidence"] > baseline["confidence"] + args.margin
        selected_name = "prompt" if use_prompt else "baseline"
        selected = prompted if use_prompt else baseline

        reference_path = find_original_transcript(audio_file)
        reference = None
        if reference_path:
            with open(reference_path, "r", encoding="utf-8-sig") as handle:
                reference = handle.read().strip()

        baseline_cer = cer_payload(reference, baseline["text"]) if reference else None
        prompt_cer = cer_payload(reference, prompted["text"]) if reference else None
        selected_cer = cer_payload(reference, selected["text"]) if reference else None
        oracle = None
        if baseline_cer and prompt_cer:
            oracle = "prompt" if prompt_cer["cer"] < baseline_cer["cer"] else "baseline"

        output_name = os.path.splitext(name)[0] + "_routed_asr.txt"
        with open(os.path.join(args.folder, output_name), "w", encoding="utf-8") as handle:
            handle.write(selected["text"])

        details.append(
            {
                "audio_file": name,
                "audio_duration_seconds": round(duration, 4),
                "reference": reference,
                "baseline": baseline,
                "prompt": prompted,
                "selected": selected_name,
                "selected_text": selected["text"],
                "oracle_selection": oracle,
                "baseline_cer": baseline_cer,
                "prompt_cer": prompt_cer,
                "routed_cer": selected_cer,
            }
        )
        print(
            f"  baseline={baseline['confidence']:.4f}, "
            f"prompt={prompted['confidence']:.4f}, selected={selected_name}"
        )

    baseline_cers = [x["baseline_cer"]["cer"] for x in details if x["baseline_cer"]]
    prompt_cers = [x["prompt_cer"]["cer"] for x in details if x["prompt_cer"]]
    routed_cers = [x["routed_cer"]["cer"] for x in details if x["routed_cer"]]
    oracle_cers = [
        min(x["baseline_cer"]["cer"], x["prompt_cer"]["cer"])
        for x in details
        if x["baseline_cer"] and x["prompt_cer"]
    ]
    router_matches_oracle = sum(
        1 for x in details if x["oracle_selection"] == x["selected"]
    )

    result = {
        "summary": {
            "total_files": len(details),
            "model_load_seconds": round(model_load_seconds, 4),
            "margin": args.margin,
            "baseline_average_cer": average(baseline_cers),
            "prompt_average_cer": average(prompt_cers),
            "routed_average_cer": average(routed_cers),
            "oracle_average_cer": average(oracle_cers),
            "prompt_selected_count": sum(1 for x in details if x["selected"] == "prompt"),
            "baseline_selected_count": sum(
                1 for x in details if x["selected"] == "baseline"
            ),
            "router_oracle_match_count": router_matches_oracle,
            "router_oracle_match_rate": router_matches_oracle / len(details),
            "average_dual_pass_seconds": average(
                [
                    x["baseline"]["inference_seconds"]
                    + x["prompt"]["inference_seconds"]
                    for x in details
                ]
            ),
        },
        "detailed_results": details,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("\n=== 路由評測完成 ===")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")
    print(f"結果已儲存至: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
