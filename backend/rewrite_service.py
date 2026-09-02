import json
import os
import re
import time
import urllib.error
import urllib.request


OLLAMA_URL = os.getenv("TAIWAN_CONTEXT_LLM_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("TAIWAN_CONTEXT_LLM_MODEL", "qwen2.5:3b")

AUDIENCE_LABELS = {
    "elder": "長輩",
    "family": "家人",
    "friend": "朋友",
    "formal": "正式場合的對象",
}
TONE_LABELS = {
    "warm": "溫暖體貼",
    "clear": "清楚直接",
    "polite": "禮貌委婉",
    "concise": "簡短自然",
}
NEGATION_MARKERS = ("不", "沒", "無", "毋", "莫", "免", "未")
ADDRESSEE_TERMS = (
    "阿嬤", "阿公", "媽媽", "爸爸", "媽", "爸", "老師", "主任",
    "叔叔", "阿姨", "伯父", "伯母", "哥哥", "姊姊", "姐姐"
)


class RewriteService:
    def __init__(self, timeout=60):
        self.timeout = timeout

    @staticmethod
    def _post_json(path, payload, timeout):
        request = urllib.request.Request(
            f"{OLLAMA_URL}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _get_json(path, timeout=5):
        with urllib.request.urlopen(f"{OLLAMA_URL}{path}", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self):
        try:
            payload = self._get_json("/api/tags")
            installed = [item.get("name", "") for item in payload.get("models", [])]
            return {
                "status": "ok",
                "provider": "ollama",
                "configured_model": OLLAMA_MODEL,
                "model_installed": any(
                    name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":")
                    for name in installed
                ),
                "installed_models": installed,
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "provider": "ollama",
                "configured_model": OLLAMA_MODEL,
                "detail": str(exc),
            }

    def _chat_json(self, system_prompt, user_prompt, temperature=0.0):
        try:
            response = self._post_json(
                "/api/chat",
                {
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": temperature, "top_p": 0.9},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                self.timeout,
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(f"無法連線本機Ollama: {exc}") from exc
        content = response.get("message", {}).get("content", "")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("本機模型未回傳有效JSON") from exc

    @staticmethod
    def _system_prompt():
        return """你是「語你傳心」臺灣語境表達助手。你的任務只限改寫措辭，不是創作或補完內容。
規則：
1. 完整保留原句事實、否定、人物、時間、地點、數字與承諾。
2. 禁止新增安慰、承諾、解決方案、建議、理由、道歉、時間或後續行動。
3. 使用繁體中文與臺灣自然說法；原句若有臺語詞，可自然保留。
4. 依對象與語氣調整措辭，但不得改變原意。
5. 若原句資訊不足，不得自行補充。
錯誤示例：原句「今天忙，不能回家吃飯」不可加入「我會想辦法解決」「下次補償你」「你放心」。
正確示例：「阿嬤，我今天工作比較忙，沒辦法回家吃飯。」
只輸出JSON，欄位必須是 rewritten_text、meaning_summary、tone、warnings。"""

    @staticmethod
    def _verification_prompt():
        return """你是嚴格的語意保真審查器。比較原句與改寫句，只判斷資訊是否一致。
只要改寫句新增任何承諾、安慰、解決方案、理由、時間、人物、地點或後續行動，就必須判定meaning_preserved為false。
語氣詞與稱謂調整可以接受，但不能形成新事實。
只輸出JSON：
{"meaning_preserved":true或false,"added_information":[],"removed_information":[],"changed_facts":[]}"""

    @staticmethod
    def _safe_fallback(original_text, audience, tone, scenario):
        """Apply only discourse-level edits using information the user supplied."""
        text = original_text.strip()
        addressee = next(
            (term for term in ADDRESSEE_TERMS if term in scenario), None
        )
        if addressee and not text.startswith(addressee):
            return f"{addressee}，{text}"
        if audience == "formal" and not text.startswith("您好"):
            return f"您好，{text}"
        if audience == "elder" and tone in {"warm", "polite"}:
            return f"想跟您說，{text}"
        return text

    def rewrite(self, original_text, audience, tone, scenario=""):
        audience_label = AUDIENCE_LABELS[audience]
        tone_label = TONE_LABELS[tone]
        user_prompt = f"""原句：{original_text}
溝通對象：{audience_label}
期望語氣：{tone_label}
情境補充：{scenario or '未提供'}

請輸出JSON：
{{"rewritten_text":"...","meaning_summary":"...","tone":"...","warnings":[]}}"""
        started = time.perf_counter()
        result = self._chat_json(self._system_prompt(), user_prompt, temperature=0.0)

        rewritten = str(result.get("rewritten_text", "")).strip()
        if not rewritten:
            raise RuntimeError("本機模型未產生改寫內容")

        warnings = [str(item) for item in result.get("warnings", [])]
        source_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", original_text))
        output_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", rewritten))
        if source_numbers != output_numbers:
            warnings.append("numbers_changed_or_missing")
        source_has_negation = any(marker in original_text for marker in NEGATION_MARKERS)
        output_has_negation = any(marker in rewritten for marker in NEGATION_MARKERS)
        if source_has_negation and not output_has_negation:
            warnings.append("negation_may_be_missing")

        verification = self._chat_json(
            self._verification_prompt(),
            f"原句：{original_text}\n改寫句：{rewritten}",
            temperature=0.0,
        )
        meaning_preserved = bool(verification.get("meaning_preserved", False))
        added_information = [str(x) for x in verification.get("added_information", [])]
        removed_information = [str(x) for x in verification.get("removed_information", [])]
        changed_facts = [str(x) for x in verification.get("changed_facts", [])]
        if not meaning_preserved:
            warnings.append("semantic_fidelity_check_failed")
        if added_information:
            warnings.append("added_information_detected")
        if removed_information:
            warnings.append("removed_information_detected")
        if changed_facts:
            warnings.append("changed_facts_detected")

        llm_candidate_safe = not warnings and meaning_preserved
        fallback_used = not llm_candidate_safe
        final_text = (
            rewritten
            if llm_candidate_safe
            else self._safe_fallback(original_text, audience, tone, scenario)
        )
        safe_to_use = llm_candidate_safe or final_text != ""
        if fallback_used:
            warnings.append("safe_fallback_applied")

        return {
            "original_text": original_text,
            "rewritten_text": final_text,
            "generated_candidate": rewritten,
            "meaning_summary": (
                str(result.get("meaning_summary", "")).strip()
                if llm_candidate_safe
                else "保留原句內容，僅調整稱謂或開頭語氣。"
            ),
            "audience": audience,
            "tone": tone,
            "scenario": scenario,
            "warnings": sorted(set(warnings)),
            "needs_confirmation": fallback_used,
            "safe_to_use": safe_to_use,
            "fallback_used": fallback_used,
            "verification": {
                "meaning_preserved": meaning_preserved,
                "added_information": added_information,
                "removed_information": removed_information,
                "changed_facts": changed_facts,
            },
            "model": OLLAMA_MODEL,
            "processing_seconds": round(time.perf_counter() - started, 4),
        }
