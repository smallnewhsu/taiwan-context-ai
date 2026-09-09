import importlib
import os
import re
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path


# ctranslate2 on Windows needs NVIDIA DLL directories before faster_whisper import.
if sys.platform == "win32":
    for package_name in ("nvidia.cudnn", "nvidia.cublas"):
        try:
            package = importlib.import_module(package_name)
            package_file = getattr(package, "__file__", None)
            if package_file:
                package_dir = os.path.dirname(package_file)
            else:
                paths = list(getattr(package, "__path__", []) or [])
                package_dir = paths[0] if paths else None
            if package_dir:
                bin_dir = os.path.join(package_dir, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
        except ImportError:
            pass

import librosa
import opencc
from faster_whisper import WhisperModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "services" / "asr" / "models"
PROMPT_PATH = (
    PROJECT_ROOT / "datasets" / "taiwan_context" / "asr_prompt_family.txt"
)
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".webm", ".ogg"}


class ASRService:
    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._converter = opencc.OpenCC("s2tw")

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if not MODEL_PATH.exists():
                        raise RuntimeError(f"找不到ASR模型: {MODEL_PATH}")
                    self._model = WhisperModel(
                        str(MODEL_PATH), device="cuda", compute_type="float16"
                    )
        return self._model

    @staticmethod
    def _clean_text(text):
        text = unicodedata.normalize("NFKC", text or "")
        text = re.sub(
            r'[,"\'。，；「」《》:：\[\]、【】〈〉（）『』…(),`&!?！;]',
            "",
            text,
        )
        return re.sub(r"\s+", "", text).strip()

    def _transcribe(self, audio, initial_prompt=""):
        model = self._load_model()
        started = time.perf_counter()
        segments, _ = model.transcribe(
            audio,
            language="zh",
            word_timestamps=False,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=True,
            initial_prompt=initial_prompt,
        )
        segments = list(segments)
        inference_seconds = time.perf_counter() - started
        text = self._clean_text(
            self._converter.convert("".join(segment.text for segment in segments))
        )

        weights = [max(segment.end - segment.start, 0.001) for segment in segments]
        total_weight = sum(weights)
        confidence = (
            sum(
                segment.avg_logprob * weight
                for segment, weight in zip(segments, weights)
            )
            / total_weight
            if total_weight
            else -999.0
        )
        return {
            "text": text,
            "confidence": float(confidence),
            "inference_seconds": round(inference_seconds, 4),
        }

    def interpret_bytes(self, content, suffix):
        suffix = suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支援的音檔格式: {suffix}")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(content)
                temp_path = handle.name

            audio, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
            audio_duration = len(audio) / float(sample_rate)
            prompt = PROMPT_PATH.read_text(encoding="utf-8-sig").strip()

            total_started = time.perf_counter()
            with self._inference_lock:
                baseline = self._transcribe(audio)
                prompted = self._transcribe(audio, prompt)
            total_seconds = time.perf_counter() - total_started

            return {
                "baseline": baseline,
                "prompt": prompted,
                "audio_duration_seconds": round(audio_duration, 4),
                "dual_pass_seconds": round(total_seconds, 4),
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
