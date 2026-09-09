import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MultimodalService:
    """Answers spoken questions using only the retained image-analysis context."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("MULTIMODAL_QA_MODEL", "qwen2.5:1.5b")

    def answer(self, question: str, image_context: dict, relationship: str = "未提供") -> dict:
        started = time.perf_counter()
        prompt = f"""你是 Taiwan Context Engine，任務是協助一般使用者理解臺灣生活語境。請直接回答使用者目前的問題，不要描述你的任務，也不要叫使用者再去回覆另一位「使用者」。

回答規則：
1. 第一個句子必須直接回答問題。若問題詢問情緒或意圖，圖片不足以證明時，先說「無法只憑這張圖片確定」，再說明較合理的可能性。
2. 將「畫面可確認的內容」與「可能的語境或意圖」分開。推測一律使用「可能、比較像、較可能」等語氣，不可當成事實。
3. visible_text 是 OCR 候選，不一定正確。confidence 低於 0.85 時不得逐字引用；即使高於 0.85，若文字不通順，也只能說「紙條大意可能是……」，不得照抄疑似錯字。
4. 可依臺灣家庭語境解釋關心、提醒、邀請或交代，但不得虛構人物身分、情緒、地點、食物種類或未出現在資料中的事件。
5. 回答應自然、溫和且可直接閱讀，避免「根據圖片提供的資訊回答」「建議您回覆使用者」等系統式套話。
6. 若問題是在問「怎麼回覆」，可提供一句簡短、可修改的建議回應；否則不要強行加入建議回應。
7. basis 只列出支持本次回答的具體線索；uncertainties 列出仍無法確認的事項。
8. 嚴格輸出 JSON：{{"answer":"2至4句的直接回答","basis":["具體線索"],"uncertainties":["無法確認事項"]}}

正確示例：
問題：「她是在生氣嗎？」
回答：「無法只憑這張紙條確定她在生氣。內容比較像是在交代飯菜放在哪裡，並提醒不要再吃泡麵，可能包含對家人飲食的關心；語氣是否不高興仍需配合前後對話判斷。」

人物關係：{relationship}
影像分析：{json.dumps(image_context, ensure_ascii=False)}
使用者問題：{question}"""
        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json", "keep_alive": "5m", "options": {"temperature": 0.1, "num_ctx": 4096}}
        req = Request(f"{self.base_url}/api/generate", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=90) as response:
                outer = json.loads(response.read().decode("utf-8"))
            result = json.loads(outer.get("response", "{}"))
        except HTTPError as exc:
            raise RuntimeError(f"語境問答模型回傳 HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"語境問答模型無法使用: {exc}") from exc
        answer = str(result.get("answer", "")).strip()
        if not answer:
            raise RuntimeError("語境問答模型未回傳答案")
        uncertainties = [str(x).strip() for x in result.get("uncertainties", []) if str(x).strip()]
        return {"answer": answer, "basis": [str(x).strip() for x in result.get("basis", []) if str(x).strip()], "uncertainties": uncertainties, "needs_confirmation": bool(uncertainties), "model": self.model, "processing_seconds": round(time.perf_counter()-started, 4)}
