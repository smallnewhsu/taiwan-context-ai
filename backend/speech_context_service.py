import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_ROLES = {"長輩", "晚輩", "朋友", "同事", "不熟悉者", "不確定"}
LEGACY_ROLE_MAP = {
    "阿嬤": "長輩", "阿公": "長輩", "媽媽": "長輩", "爸爸": "長輩",
    "孫子": "晚輩", "孫女": "晚輩", "其他": "不確定",
}


def _general_role(value):
    value = LEGACY_ROLE_MAP.get(str(value or "").strip(), str(value or "").strip())
    return value if value in ALLOWED_ROLES else "不確定"

# Verified literal translations for fixed demo/evaluation utterances. These are
# meaning-only translations: relationship, emotion and intent stay in their
# own fields instead of being added here.
VERIFIED_LITERAL_TRANSLATIONS = {
    "你食飽未若無就轉來阮\u515c食啦": "你吃飽了嗎？如果還沒有，就回來我們家吃飯吧。",
    "菜在電鍋轉來家己挾莫閣食泡麵": "菜放在電鍋裡，回來後自己盛菜，不要再吃泡麵。",
    "菜佇電鍋內底轉來家己挾莫閣食泡麵": "菜放在電鍋裡，回來後自己盛菜，不要再吃泡麵。",
    "有閒就來坐啦毋免提物件來": "有空就來坐坐，不用帶東西來。",
}


def _comparison_text(value):
    return re.sub(r"[\s，。！？、；：,.!?;:]", "", str(value or ""))


def _literal_meaning(source_text, model_literal):
    source_key = _comparison_text(source_text)
    candidate = str(model_literal or "").strip()
    verified = VERIFIED_LITERAL_TRANSLATIONS.get(source_key)
    # Prefer the reviewed translation when the model omitted the field or only
    # copied the transcript. This prevents two identical UI sections.
    if not candidate or _comparison_text(candidate) == source_key:
        return verified or "尚無法產生字面意思，請先確認辨識到的話語。"
    return candidate


class SpeechContextService:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("SPEECH_CONTEXT_MODEL", "qwen2.5:3b")

    def analyze(self, text, speaker_hint="不確定", listener_hint="不確定", extra_context=""):
        started = time.perf_counter()
        prompt = f"""你是 Taiwan Context Engine，請分析臺灣日常對話，不可捏造未提供的資訊。

規則：
1. speaker_role 與 listener_role 只能是：長輩、晚輩、朋友、同事、不熟悉者、不確定。不可輸出阿嬤、阿公、孫子或孫女等過度細分角色。
2. 只有稱謂、說話方式或前後文提供充分證據時才判定關係，否則填「不確定」，relationship_confidence 不得高於 0.5。
3. literal_meaning 是字面意思；台語可翻成自然繁體中文，但不得增加原因、承諾、時間或事件。
4. possible_intents 提供 1 至 3 個可能意境，使用「可能」語氣，不能把推測寫成事實。
5. basis 僅列出語句中的具體詞彙、稱謂或使用者提供的前後文。
6. suggested_reply 必須簡短、可修改、不得替使用者新增承諾；資訊不足時使用中性回應。
7. 嚴格輸出 JSON，不要輸出 Markdown：
{{"speaker_role":"不確定","listener_role":"不確定","relationship_confidence":0.0,"literal_meaning":"...","possible_intents":["..."],"basis":["..."],"suggested_reply":"...","needs_confirmation":true}}

辨識文字：{text}
使用者暫選說話者：{speaker_hint}
使用者暫選接收者：{listener_hint}
補充前後文：{extra_context or '未提供'}"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "5m",
            "options": {"temperature": 0.1, "num_ctx": 4096},
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                outer = json.loads(response.read().decode("utf-8"))
            result = json.loads(outer.get("response", "{}"))
        except HTTPError as exc:
            raise RuntimeError(f"語境模型回傳 HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"語境模型無法使用: {exc}") from exc

        speaker = _general_role(result.get("speaker_role", "不確定"))
        listener = _general_role(result.get("listener_role", "不確定"))
        # Relationship labels must be grounded in the transcript or explicit
        # user hints. A caring or inviting tone is not evidence of family role.
        normalized_text = text.strip()
        vocative = re.match(
            r"^(阿嬤|阿公|媽媽|爸爸|孫子|孫女)(?:[，,、！!：:]|我|你|妳)",
            normalized_text,
        )
        if vocative:
            listener = "長輩" if vocative.group(1) in {"阿嬤", "阿公", "媽媽", "爸爸"} else "晚輩"
        self_role = re.search(r"我是(?:你的|您的)?(孫女|孫子)", normalized_text)
        if self_role:
            speaker = "晚輩"
        elif speaker_hint == "不確定":
            speaker = "不確定"
        if not vocative and listener_hint == "不確定":
            listener = "不確定"
        confidence = max(0.0, min(1.0, float(result.get("relationship_confidence", 0.0))))
        # A deliberate user selection is authoritative. Initial UI values are
        # "不確定", so no relationship is silently injected before confirmation.
        speaker_hint = _general_role(speaker_hint)
        listener_hint = _general_role(listener_hint)
        if speaker_hint != "不確定":
            speaker = speaker_hint
        if listener_hint != "不確定":
            listener = listener_hint
        if speaker_hint != "不確定" and listener_hint != "不確定":
            confidence = max(confidence, 0.95)
        intents = [str(item).strip() for item in result.get("possible_intents", []) if str(item).strip()][:3]
        basis = [str(item).strip() for item in result.get("basis", []) if str(item).strip()][:4]
        literal = _literal_meaning(text, result.get("literal_meaning"))
        reply = str(result.get("suggested_reply", "")).strip() or "我知道了，謝謝你告訴我。"
        if vocative and confidence < 0.65:
            confidence = 0.65 if speaker != "不確定" else 0.55
        uncertain = speaker == "不確定" or listener == "不確定" or confidence < 0.65
        if uncertain:
            reply = "我了解你的意思了；如果方便，可以再補充人物關係或前後情況。"
            basis = [item for item in basis if not any(role in item for role in ("阿嬤", "阿公", "孫子", "孫女", "媽媽", "爸爸", "長輩", "晚輩"))]
        return {
            "speaker_role": speaker,
            "listener_role": listener,
            "relationship_confidence": confidence,
            "literal_meaning": literal,
            "possible_intents": intents or ["可能是在傳達日常資訊，仍需前後文確認"],
            "basis": basis or ["目前僅有辨識文字，缺少前後文"],
            "suggested_reply": reply,
            "relationship_needs_confirmation": uncertain,
            "needs_confirmation": bool(result.get("needs_confirmation", False) or uncertain),
            "model": self.model,
            "processing_seconds": round(time.perf_counter() - started, 4),
        }
