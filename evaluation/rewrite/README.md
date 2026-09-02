# 語你傳心 v0.1 正式評測

本套件只評估已鎖定的 v0.1，不修改 Prompt、模型或安全規則。12 個案例涵蓋家庭、長輩、朋友、職場與公告情境，以及數字、時間、否定、因果、不確定性、取消、金額限制和臺灣台語混合句。案例使用 API 支援的對象代碼：`elder`、`family`、`friend`、`formal`。

## 執行前

1. Ollama 已啟動，且 `qwen2.5:3b` 可用。
2. 後端 API 已在 `http://127.0.0.1:8000` 執行。
3. 將本套件內的 `evaluation` 與 `tools` 目錄合併至專案根目錄。

## 執行

在 `C:\taiwan-context-ai` 執行：

```bat
python tools\evaluate_rewrite.py
```

每個案例通常會呼叫本機模型進行改寫與語意驗證，請等待全部 12 案例完成。

輸出位於：

```text
evaluation\rewrite\results\v0.1\summary.json
evaluation\rewrite\results\v0.1\rewrite_evaluation_results.json
evaluation\rewrite\results\v0.1\human_review_template.csv
```

## 指標解讀

- `automatic_pass_rate`：最終輸出通過必要詞、禁止新增內容、數字與否定檢查的比例。
- `llm_candidate_accepted_rate`：LLM 原始候選通過服務內語意保真檢查的比例。
- `safe_fallback_rate`：候選被攔截後，改用保守式安全改寫的比例。
- `needs_confirmation_count`：仍需使用者確認的案例數。
- `mean/median/p95_processing_seconds`：後端回報的處理延遲。

自動通過不等於自然流暢。請開啟 `human_review_template.csv`，以 1–5 分人工評估：

- `fidelity_score_1_5`：是否完整保留原意且未新增承諾、原因或行動。
- `audience_fit_score_1_5`：是否符合指定對象與情境。
- `naturalness_score_1_5`：是否自然、像真人訊息。

建議先完成評測與人工審查，再決定是否提交結果或調整 v0.2。
