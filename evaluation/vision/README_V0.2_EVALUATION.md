# 視界有解 v0.2 評測

本評測分為兩部分：

1. 使用既有 `tools/evaluate_vision.py` 重新測試目前正式影像分析管線。
2. 使用 `evaluation/scripts/evaluate_vision_multimodal.py` 測試圖片分析後接文字或語音追問。

## 檔案配置

將檔案放入主專案後，確認12張圖片位於：

```text
evaluation/vision/V01.jpg
...
evaluation/vision/V12.jpg
```

## 執行影像分析評測

```bat
python tools\evaluate_vision.py ^
  --cases evaluation\vision\test_cases.json ^
  --output evaluation\vision\results\v0.2_gemma3_4b_gpu ^
  --url http://127.0.0.1:8000/vision/analyze ^
  --timeout 240
```

## 執行影像後接文字追問

```bat
python evaluation\scripts\evaluate_vision_multimodal.py ^
  --cases evaluation\vision\test_cases_multimodal_v0.2.json ^
  --images evaluation\vision ^
  --output evaluation\vision\results\v0.2_multimodal_gemma3_4b_gpu ^
  --api-url http://127.0.0.1:8000 ^
  --timeout 240 ^
  --delay 0.2
```

## 語音追問

測試案例預設使用文字追問，以固定問題內容並確保結果可重現。若要測試語音追問，在個案中加入：

```json
"audio_file": "M01_question.m4a"
```

並將音檔放入 `evaluation/vision/`。評測程式會自動改用 `/vision/follow-up`，同時記錄ASR與多模態回答。

## 自動指標

- 流程完成率
- 回答關鍵詞召回率
- 禁止資訊未出現率（hallucination-free rate）
- 不確定性處理正確率
- 自動通過率
- 排除暖機後的平均、median及P95端到端延遲

自動關鍵詞通過不等同完整語意正確，仍須填寫 `human_review_template.csv` 進行人工複核。
