import json
import os
import re
import time
from urllib import request as urlrequest


class TaigiService:
    """Meaning-preserving Mandarin to Taiwanese Taigi (Han characters)."""

    SAFE_PHRASES = {
        "我今天無法回家吃飯": "我今仔日無法度轉去厝裡食飯。",
        "我今天不能回家吃飯": "我今仔日無法度轉去厝裡食飯。",
        "你吃飽了嗎如果還沒有就回來我們家吃": "你食飽未？若猶未，就轉來阮兜食。",
    }

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("TAIGI_MODEL", "qwen2.5:3b")

    @staticmethod
    def _key(text: str) -> str:
        return re.sub(r"[\s，。！？、,.!?；;：:]", "", text)

    def convert(self, original_text: str, audience: str, scenario: str = "") -> dict:
        started = time.perf_counter()
        known = self.SAFE_PHRASES.get(self._key(original_text))
        if known:
            return self._result(original_text, known, started, "verified_phrase", [])

        prompt = f"""你是臺灣台語文字轉寫員。請把原句直接轉成自然、可讀的台語漢字。
規則：只轉換語言，不新增原因、承諾、人物、時間或行程；數字、否定、限制與專有名詞必須完整保留；禁止使用中國方言用詞；只輸出 JSON。
JSON 格式：{{"taigi_text":"..."}}
溝通對象：{audience}
補充情境：{scenario or '無'}
原句：{original_text}"""
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.1}}, ensure_ascii=False).encode()
        req = urlrequest.Request(f"{self.base_url}/api/generate", data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlrequest.urlopen(req, timeout=60) as response:
                outer = json.loads(response.read().decode("utf-8"))
            candidate = json.loads(outer.get("response", "{}")) .get("taigi_text", "").strip()
        except Exception as exc:
            raise RuntimeError(f"台語模型無法使用: {exc}") from exc
        if not candidate:
            raise RuntimeError("台語模型未回傳內容")
        warnings = self._safety_warnings(original_text, candidate)
        final = original_text if warnings else candidate
        return self._result(original_text, final, started, "ollama", warnings, candidate)

    def _safety_warnings(self, source: str, candidate: str) -> list[str]:
        warnings = []
        source_numbers = re.findall(r"\d+", source)
        if source_numbers and source_numbers != re.findall(r"\d+", candidate):
            warnings.append("numbers_changed_or_missing")
        source_negative = any(term in source for term in ("不", "不能", "無法", "不要", "未"))
        target_negative = any(term in candidate for term in ("不", "毋", "無", "袂", "莫", "未"))
        if source_negative and not target_negative:
            warnings.append("negation_changed_or_missing")
        return warnings

    def _result(self, source, text, started, mode, warnings, candidate=""):
        return {"original_text": source, "taigi_text": text, "generated_candidate": candidate or text, "mode": mode, "warnings": warnings, "needs_confirmation": bool(warnings), "model": self.model, "processing_seconds": round(time.perf_counter() - started, 4)}
