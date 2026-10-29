#!/usr/bin/env python3
"""Merge a validated Taiwan Tongues PEFT LoRA adapter into the HF base model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Taiwan Tongues LoRA adapter")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\models\Taiwan-Tongues-ASR-CE-pretrained-v1.0"
        ),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\adapters\taigi_lora_v3_3000s300"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\models\taigi_lora_v3_merged_hf"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Use a fresh directory to preserve model provenance."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = args.adapter / "adapter_model.safetensors"
    adapter_config = args.adapter / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError("Adapter model or adapter_config.json is missing")

    print("Loading FP16 base model on CPU...")
    base = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    print("Loading validated LoRA adapter...")
    peft_model = PeftModel.from_pretrained(
        base,
        args.adapter,
        is_trainable=False,
        local_files_only=True,
    )
    print("Merging adapter into base weights...")
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language="zh",
        task="transcribe",
        local_files_only=True,
    )
    processor.save_pretrained(args.output_dir)

    model_files = sorted(args.output_dir.glob("*.safetensors"))
    if not model_files:
        raise RuntimeError("Merged model weights were not saved")
    manifest = {
        "model": "Taiwan Tongues ASR CE Taigi LoRA v3 merged",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "adapter_config_sha256": sha256(adapter_config),
        "adapter_weights_sha256": sha256(adapter_weights),
        "dtype": "float16",
        "merge_method": "PEFT merge_and_unload safe_merge=True",
        "merged_weights": [
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in model_files
        ],
    }
    manifest_path = args.output_dir / "merge_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== LoRA merge complete ===")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Merged weight files: {len(model_files)}")


if __name__ == "__main__":
    main()
