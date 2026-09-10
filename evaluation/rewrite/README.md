# 語你傳心安全改寫評測

12個案例涵蓋家庭、長輩、朋友、職場與公告情境，以及數字、時間、否定、因果、不確定性、取消、金額限制和台語混合句。案例使用API支援的對象代碼：`elder`、`family`、`friend`、`formal`。

## 執行前

1. Ollama 已啟動，且 `qwen2.5:1.5b` 可用。
2. 後端 API 已在 `http://127.0.0.1:8000` 執行。

## 執行

在 `C:\taiwan-context-ai` 執行：

```bat
python tools\evaluate_rewrite.py --output-dir evaluation\rewrite\results\v0.3_qwen1.5b_gpu
```

每個案例通常會呼叫本機模型進行改寫與語意驗證，請等待全部 12 案例完成。

輸出位於：

```text
evaluation\rewrite\results\v0.3_qwen1.5b_gpu\summary.json
evaluation\rewrite\results\v0.3_qwen1.5b_gpu\rewrite_evaluation_results.json
evaluation\rewrite\results\v0.3_qwen1.5b_gpu\human_review_template.csv
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

不同模型版本必須輸出至不同結果目錄。完成自動評測後仍須進行人工審查，不得把Qwen2.5 3B的人工評分直接套用至1.5B輸出。
