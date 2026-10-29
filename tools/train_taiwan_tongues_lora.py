#!/usr/bin/env python3
"""Memory-conscious LoRA pilot training for Taiwan Tongues ASR CE.

Designed for a 6 GB RTX 4050 laptop GPU.  The base Whisper model is loaded in
INT8 and remains frozen; only small LoRA adapters on q_proj/v_proj are trained.
The script reads the ADI-compatible TSV corpus produced by
prepare_adi_asr_webdataset.py and saves only the adapter plus processor files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


LANGUAGE_DATASETS = {
    "taigi": "train_ds_taigi",
    "hakka": "train_ds_hakka",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taiwan Tongues ASR CE INT8 LoRA pilot training"
    )
    parser.add_argument("--language", choices=sorted(LANGUAGE_DATASETS), required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            r"C:\taiwan-context-training\models\Taiwan-Tongues-ASR-CE-pretrained-v1.0"
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(r"C:\taiwan-context-training\sample_corpus"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\taiwan-context-training\adapters"),
    )
    parser.add_argument("--pilot-samples", type=int, default=200)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output folder name; defaults to <language>_pilot_lora",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"TSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"path", "sentence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"TSV must contain path and sentence columns: {path}")
    return rows


class SpeechDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        clips_dir: Path,
        processor: WhisperProcessor,
    ) -> None:
        self.rows = rows
        self.clips_dir = clips_dir
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        audio_path = self.clips_dir / row["path"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(row["sentence"]).input_ids
        return {
            "input_features": features,
            "labels": labels,
            "audio_path": str(audio_path),
        }


@dataclass
class SpeechCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        inputs = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        label_inputs = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_inputs, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if labels.shape[1] and (
            labels[:, 0] == self.decoder_start_token_id
        ).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def deterministic_subset(
    rows: list[dict[str, str]], count: int, seed: int
) -> list[dict[str, str]]:
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    if count > 0:
        selected = selected[:count]
    return selected


@torch.no_grad()
def validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float | None:
    if not len(loader):
        return None
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this pilot requires an NVIDIA GPU")
    if args.max_steps < 1 or args.gradient_accumulation < 1:
        raise ValueError("max-steps and gradient-accumulation must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset_name = LANGUAGE_DATASETS[args.language]
    dataset_dir = args.corpus_dir / dataset_name
    clips_dir = dataset_dir / "clips"
    train_rows = deterministic_subset(
        read_tsv(dataset_dir / "train.tsv"), args.pilot_samples, args.seed
    )
    validation_rows = deterministic_subset(
        read_tsv(dataset_dir / "validated.tsv"),
        args.validation_samples,
        args.seed + 1,
    )
    if not train_rows:
        raise ValueError("No training rows selected")

    print(f"Language: {args.language}")
    print(f"Train pilot rows: {len(train_rows)}")
    print(f"Validation rows: {len(validation_rows)}")
    print(f"Base model: {args.model_path}")

    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language="zh",
        task="transcribe",
        local_files_only=True,
    )
    quantization = BitsAndBytesConfig(load_in_8bit=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    device = torch.device("cuda:0")

    collator = SpeechCollator(processor, model.config.decoder_start_token_id)
    train_loader = DataLoader(
        SpeechDataset(train_rows, clips_dir, processor),
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        SpeechDataset(validation_rows, clips_dir, processor),
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.perf_counter()
    data_iterator = iter(train_loader)

    for step in range(1, args.max_steps + 1):
        micro_losses: list[float] = []
        for _ in range(args.gradient_accumulation):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation
            loss.backward()
            micro_losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_loss = sum(micro_losses) / len(micro_losses)
        if not math.isfinite(step_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {step_loss}")
        losses.append(step_loss)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"step={step}/{args.max_steps} loss={step_loss:.4f} "
            f"gpu_allocated={allocated:.2f}GB gpu_reserved={reserved:.2f}GB"
        )

    elapsed = time.perf_counter() - started
    val_loss = validation_loss(model, validation_loader, device)
    run_name = args.run_name or f"{args.language}_pilot_lora"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    metrics = {
        "language": args.language,
        "base_model": str(args.model_path),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "first_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "minimum_train_loss": min(losses),
        "validation_loss": val_loss,
        "elapsed_seconds": elapsed,
        "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "seed": args.seed,
        "run_name": run_name,
    }
    (output_dir / "pilot_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== Pilot training complete ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Adapter output: {output_dir}")


if __name__ == "__main__":
    main()
