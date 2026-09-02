import base64
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class VisionService:
    """Local, privacy-preserving image interpretation through Ollama."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: str = "http://127.0.0.1:11434",
        max_edge: int = 768,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.max_edge = max_edge

    def health(self) -> dict[str, Any]:
        installed_models: list[str] = []
        try:
            payload = self._request_json("GET", "/api/tags")
            installed_models = [
                item.get("name", "") for item in payload.get("models", [])
            ]
            installed = any(
                name == self.model or name.startswith(f"{self.model}:")
                for name in installed_models
            )
            return {
                "status": "ok",
                "provider": "ollama",
                "configured_model": self.model,
                "model_installed": installed,
                "max_image_edge": self.max_edge,
                "installed_models": installed_models,
            }
        except RuntimeError as exc:
            return {
                "status": "unavailable",
                "provider": "ollama",
                "configured_model": self.model,
                "model_installed": False,
                "detail": str(exc),
            }

    def analyze_bytes(
        self, content: bytes, suffix: str, filename: str = "image"
    ) -> dict[str, Any]:
        suffix = suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"不支援的圖片格式。允許格式: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        if not content:
            raise ValueError("圖片內容為空")

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="taiwan_vision_") as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / f"source{suffix}"
            resized_path = temp_path / "resized.jpg"
            source_path.write_bytes(content)
            original_size = self._image_size(source_path)
            self._resize_image(source_path, resized_path)
            processed_size = self._image_size(resized_path)
            encoded_image = base64.b64encode(resized_path.read_bytes()).decode("ascii")

        response = self._request_json(
            "POST",
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
                "keep_alive": "5m",
                "format": self._response_schema(),
                "options": {"temperature": 0.1, "num_ctx": 8192},
                "messages": [
                    {
                        "role": "user",
                        "content": self._prompt(),
                        "images": [encoded_image],
                    }
                ],
            },
            timeout=180,
        )

        message = response.get("message", {}).get("content", "")
        try:
            analysis = json.loads(message)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("視覺模型未回傳有效的結構化結果") from exc

        analysis = self._sanitize_analysis(analysis)
        warnings = self._safety_warnings(analysis)
        elapsed = round(time.perf_counter() - started, 4)
        return {
            "filename": filename,
            "model": self.model,
            "original_size": original_size,
            "processed_size": processed_size,
            "processing_seconds": elapsed,
            "analysis": analysis,
            "needs_confirmation": bool(warnings),
            "warnings": warnings,
            "privacy": {
                "processed_locally": True,
                "image_persisted": False,
            },
        }

    def _resize_image(self, source: Path, destination: Path) -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("找不到 ffmpeg，無法安全縮放圖片")
        scale = (
            f"scale='if(gt(iw,ih),min(iw,{self.max_edge}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{self.max_edge}))'"
        )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vf", scale, "-frames:v", "1",
            "-update", "1", str(destination),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
        if completed.returncode != 0 or not destination.exists():
            detail = completed.stderr.strip() or "未知圖片解碼錯誤"
            raise ValueError(f"圖片無法解碼或縮放: {detail}")

    @staticmethod
    def _image_size(path: Path) -> dict[str, int]:
        if not shutil.which("ffprobe"):
            return {"width": 0, "height": 0}
        command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        if completed.returncode != 0:
            raise ValueError("無法讀取圖片尺寸")
        streams = json.loads(completed.stdout).get("streams", [])
        if not streams:
            raise ValueError("檔案中沒有可辨識的圖片")
        return {
            "width": int(streams[0].get("width", 0)),
            "height": int(streams[0].get("height", 0)),
        }

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 10,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.ollama_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama 回傳 HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"無法連線至 Ollama: {exc}") from exc

    @staticmethod
    def _prompt() -> str:
        return """
你是「視界有解」的臺灣生活影像輔助模型。只根據圖片中可見證據回答。
請使用繁體中文並嚴格輸出指定 JSON，不要加入 Markdown。

規則：
1. observations 只放可直接看見的物件或環境，不推測身分、地點或用途。
2. visible_text 只抄錄清楚可讀的文字；模糊文字不得補字，每項給 0 到 1 的 confidence。
3. context_inferences 只能是「可能」情境，每項必須寫出圖片中的 basis，不能把推測寫成事實。
4. 不得僅因街景、語言或建築就斷定在臺灣，也不得虛構店名、地名、人物關係或事件。
5. uncertainties 列出重要但無法從圖片確認的資訊。
6. 若看不清楚，寧可回報不確定，不要猜測。
""".strip()

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        confidence_item = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["text", "confidence"],
        }
        inference_item = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "basis": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["description", "basis", "confidence"],
        }
        return {
            "type": "object",
            "properties": {
                "observations": {"type": "array", "items": confidence_item},
                "visible_text": {"type": "array", "items": confidence_item},
                "context_inferences": {"type": "array", "items": inference_item},
                "uncertainties": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "observations", "visible_text", "context_inferences", "uncertainties"
            ],
        }

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 3)
        except (TypeError, ValueError):
            return 0.0

    def _sanitize_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        observations = []
        for item in analysis.get("observations", []):
            text = str(item.get("text", "")).strip()
            if text:
                observations.append({
                    "text": text,
                    "confidence": self._confidence(item.get("confidence")),
                })

        visible_text = []
        for item in analysis.get("visible_text", []):
            text = str(item.get("text", "")).strip()
            if text:
                visible_text.append({
                    "text": text,
                    "confidence": self._confidence(item.get("confidence")),
                })

        inferences = []
        for item in analysis.get("context_inferences", []):
            description = str(item.get("description", "")).strip()
            basis = str(item.get("basis", "")).strip()
            if description:
                inferences.append({
                    "description": description,
                    "basis": basis,
                    "confidence": self._confidence(item.get("confidence")),
                })

        uncertainties = [
            str(item).strip() for item in analysis.get("uncertainties", [])
            if str(item).strip()
        ]
        return {
            "observations": observations,
            "visible_text": visible_text,
            "context_inferences": inferences,
            "uncertainties": uncertainties,
        }

    @staticmethod
    def _safety_warnings(analysis: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if any(item["confidence"] < 0.85 for item in analysis["visible_text"]):
            warnings.append("low_confidence_ocr_requires_confirmation")
        if analysis["context_inferences"]:
            warnings.append("context_is_inference_not_fact")
        if any(
            item["confidence"] < 0.75 or not item["basis"]
            for item in analysis["context_inferences"]
        ):
            warnings.append("low_confidence_context_requires_confirmation")
        if any(item["confidence"] < 0.65 for item in analysis["observations"]):
            warnings.append("low_confidence_observation")
        return warnings
