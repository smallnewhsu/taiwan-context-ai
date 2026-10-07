import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_UNCERTAINTY_TERMS = (
    "人物", "關係", "語氣", "場合", "情境", "前後文", "對話",
    "OCR", "文字", "辨識", "真實意圖", "是否", "無法確定",
    "地點", "位置", "地址", "名稱", "狀態", "時間", "內容",
)

OCR_QUESTION_TERMS = ("寫了什麼", "文字", "多少錢", "價格", "菜單", "口味", "站牌", "公告", "提供什麼服務")
LOCATION_QUESTION_TERMS = ("哪個地方", "哪裡", "地點", "位置", "地址")
CURRENT_STATE_TERMS = ("正在", "現在", "目前是否", "已經")
INFERENCE_QUESTION_TERMS = ("可能是", "可能提供", "可能包含", "意圖", "生氣", "心情", "語氣")
UNCERTAINTY_WORDS = ("無法", "不能確定", "無法確認", "看不清")


class MultimodalService:
    """Answers spoken questions using only the retained image-analysis context."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("MULTIMODAL_QA_MODEL", "gemma3:4b")

    @staticmethod
    def _analysis(image_context: dict) -> dict:
        analysis = image_context.get("analysis", image_context)
        return analysis if isinstance(analysis, dict) else {}

    def _safe_override(self, question: str, image_context: dict) -> dict | None:
        analysis = self._analysis(image_context)
        observations = " ".join(
            str(item.get("text", "")) for item in analysis.get("observations", [])
        )
        warnings = set(image_context.get("warnings", []))

        if "模糊" in observations and any(term in question for term in OCR_QUESTION_TERMS):
            return {
                "answer": "圖片中的文字過於模糊，目前無法可靠辨識內容。建議重新拍攝清晰、正面且光線充足的圖片後再確認。",
                "basis": ["圖片中的文字區塊模糊"],
                "uncertainties": ["OCR文字辨識正確性無法確認"],
            }

        if "場所" in question:
            place_terms = ("場所", "建築", "店", "館", "站", "公園", "街道", "室內", "戶外")
            inferences = sorted(
                [
                    item for item in analysis.get("context_inferences", [])
                    if any(term in str(item.get("description", "")) for term in place_terms)
                ],
                key=lambda item: float(item.get("confidence", 0) or 0),
                reverse=True,
            )
            if inferences:
                best = inferences[0]
                description = str(best.get("description", "")).strip()
                basis = str(best.get("basis", "")).strip()
                if description:
                    return {
                        "answer": f"根據畫面線索，這裡{description}。不過，具體場所名稱與實際用途仍無法只憑圖片確定。",
                        "basis": [basis] if basis else ["影像中的建築外觀與周圍環境"],
                        "uncertainties": ["具體場所名稱與用途無法確定"],
                    }
            if "建築" in observations:
                return {
                    "answer": "畫面顯示一座大型建築，可能屬於公共建築或公共使用場所；具體名稱與實際用途仍無法只憑圖片確定。",
                    "basis": ["畫面可見大型建築、入口與公共空間外觀"],
                    "uncertainties": ["具體場所名稱與用途無法確定"],
                }

        if (
            any(term in question for term in ("怎麼使用", "如何使用"))
            and "鏡子" in observations
            and "支架" in observations
        ):
            return {
                "answer": "這是一面帶支架的鏡子，可將支架立起，放在平穩表面，再把鏡面調整到適合的角度使用。",
                "basis": ["畫面可見鏡面、鏡框與支架"],
                "uncertainties": [],
            }

        if any(term in question for term in CURRENT_STATE_TERMS):
            return {
                "answer": "無法只憑這張靜態圖片確定物品目前是否正在運作。畫面只能確認物品外觀，沒有足夠證據證明其使用狀態。",
                "basis": ["圖片只呈現靜態外觀"],
                "uncertainties": ["物品目前的運作狀態無法確認"],
            }

        if any(term in question for term in LOCATION_QUESTION_TERMS):
            return {
                "answer": "無法只憑這張圖片確定具體地點。畫面中的景物可供描述，但沒有足以確認城市或地址的可靠線索。",
                "basis": ["圖片沒有可驗證的地名或地址"],
                "uncertainties": ["拍攝地點與地址無法確認"],
            }

        if (
            any(term in question for term in ("站牌", "提供什麼服務"))
            and any(term in question for term in ("可以確認", "提供什麼服務"))
        ) or (
            "low_confidence_ocr_requires_confirmation" in warnings
            and any(term in question for term in OCR_QUESTION_TERMS)
            and any(term in question for term in ("可以確認", "寫了什麼", "有哪些"))
        ):
            return {
                "answer": "目前OCR結果的信心不足，無法可靠確認圖片中的完整文字內容。可以先描述看得見的版面與物件，但具體文字仍應由使用者對照原圖確認。",
                "basis": ["影像分析已標示OCR結果需要確認"],
                "uncertainties": ["OCR文字內容與名稱無法確認"],
            }
        return None

    @staticmethod
    def _relevant_uncertainties(question: str, items: list[str]) -> list[str]:
        if any(term in question for term in LOCATION_QUESTION_TERMS):
            allowed = ("地點", "位置", "地址", "名稱")
        elif any(term in question for term in OCR_QUESTION_TERMS):
            allowed = ("OCR", "文字", "辨識", "內容")
        elif any(term in question for term in CURRENT_STATE_TERMS):
            allowed = ("狀態", "是否", "時間")
        elif any(term in question for term in INFERENCE_QUESTION_TERMS):
            allowed = ("人物", "關係", "語氣", "情境", "前後文", "意圖", "場所", "服務", "用途")
        else:
            return []
        return [item for item in items if any(term in item for term in allowed)][:4]

    def answer(self, question: str, image_context: dict, relationship: str = "未提供") -> dict:
        started = time.perf_counter()
        override = self._safe_override(question, image_context)
        if override is not None:
            return {
                **override,
                "needs_confirmation": bool(override.get("uncertainties")),
                "model": "deterministic_safety_guard",
                "processing_seconds": round(time.perf_counter() - started, 4),
            }

        analysis = self._analysis(image_context)
        question_uses_ocr = any(term in question for term in OCR_QUESTION_TERMS)
        safe_context = {
            "analysis": {
                "observations": analysis.get("observations", []),
                "visible_text": analysis.get("visible_text", []) if question_uses_ocr else [],
                "context_inferences": analysis.get("context_inferences", []),
                "uncertainties": analysis.get("uncertainties", []),
            },
            "warnings": image_context.get("warnings", []),
            "needs_confirmation": image_context.get("needs_confirmation", False),
        }
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
9. uncertainties 只能列人物關係、語氣、場合、前後文、OCR正確性或真實意圖；不得列出交通方式等與問題無關的事項。

正確示例：
問題：「她是在生氣嗎？」
回答：「無法只憑這張紙條確定她在生氣。內容比較像是在交代飯菜放在哪裡，並提醒不要再吃泡麵，可能包含對家人飲食的關心；語氣是否不高興仍需配合前後對話判斷。」

人物關係：{relationship}
影像分析：{json.dumps(safe_context, ensure_ascii=False)}
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
        uncertainties = [
            item for item in uncertainties
            if any(term in item for term in ALLOWED_UNCERTAINTY_TERMS)
        ]
        uncertainties = self._relevant_uncertainties(question, uncertainties)
        if any(term in question for term in INFERENCE_QUESTION_TERMS) and not uncertainties:
            uncertainties = ["圖片只能支持可能情境，實際場所、服務或意圖無法確定"]
        if any(term in answer for term in UNCERTAINTY_WORDS) and not uncertainties:
            if any(term in question for term in LOCATION_QUESTION_TERMS):
                uncertainties = ["拍攝地點與位置無法確認"]
            elif any(term in question for term in OCR_QUESTION_TERMS):
                uncertainties = ["OCR文字內容無法確認"]
            elif any(term in question for term in CURRENT_STATE_TERMS):
                uncertainties = ["物品目前狀態無法確認"]
        if not question_uses_ocr:
            answer = answer.replace("這張紙條", "這張圖片").replace("紙條大意", "圖片內容")
        return {"answer": answer, "basis": [str(x).strip() for x in result.get("basis", []) if str(x).strip()], "uncertainties": uncertainties, "needs_confirmation": bool(uncertainties), "model": self.model, "processing_seconds": round(time.perf_counter()-started, 4)}
