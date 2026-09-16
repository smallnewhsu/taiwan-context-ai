# 語你傳心評測

本模組檢查表達調整是否保留原意、數字、否定、限制條件、時間及在地詞彙，並記錄模型候選、安全回退、人工複核與推論時間。

## 正式版本

|版本|用途|結果|
|---|---|---|
|v0.6.2|36筆擴充開發集|自動安全通過率100%；安全回退率72.22%|
|v0.6.3|12筆凍結獨立測試|自動83.33%；人工原始輸出91.67%；修正後安全完成100%|

正式競賽版本使用Gemma 3 4B。舊模型資料夾只保留原始JSON、CSV與工作簿供版本追溯，不作為正式成果。

## 執行方式

```bat
python tools\evaluate_rewrite.py --cases evaluation\rewrite\test_cases_v0.4_expanded.json --api-url http://127.0.0.1:8000/expression/rewrite --output-dir evaluation\rewrite\results\v0.6.2_expanded_gemma3_4b_gpu
python tools\evaluate_rewrite.py --cases evaluation\rewrite\test_cases_v0.5_holdout.json --api-url http://127.0.0.1:8000/expression/rewrite --output-dir evaluation\rewrite\results\v0.6.3_holdout_gemma3_4b_gpu
```

## 指標定義

- 自動安全通過：最終輸出符合測試案例的必要資訊與限制條件。
- 模型候選直接採用：候選未觸發安全回退。
- 安全回退：候選可能偏離原意，系統改用保守輸出並要求確認。
- 人工複核通過：三項人工分數均至少4分。
- 修正後安全完成：原始輸出未通過，但經人工修正與確認後可使用。

安全回退不是API失敗；自動通過也不代表自然度或一般化能力。
