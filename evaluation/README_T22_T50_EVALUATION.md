# T22～T50 擴充 ASR 評測

本套件直接使用專案既有的 `services/asr/evaluate_asr.py` 與已建立的標準逐字稿，不會重建或覆寫逐字稿。

## 放置方式

將 ZIP 解壓縮到：

`C:\taiwan-context-ai`

確認程式位於：

`C:\taiwan-context-ai\evaluation\scripts\evaluate_asr_extended.py`

音檔與逐字稿應位於：

`C:\taiwan-context-ai\evaluation\asr_extended`

檔名範例：`T22.m4a`、`T22_reference.txt`，依序至 `T50.m4a`、`T50_reference.txt`。

## 執行

先關閉正在占用 GPU 的本機模型或 API，再於 CMD 執行：

```bat
cd /d C:\taiwan-context-ai
call services\asr\asr_api\Scripts\activate.bat
python evaluation\scripts\evaluate_asr_extended.py
```

程式會先嚴格檢查 29 個音檔與 29 個非空白標準逐字稿，再開始 GPU 推論。

## 輸出

結果會寫入 `evaluation\asr_extended\`：

- `asr_extended_results_gpu_29.json`：逐筆結果、整體統計及分語言統計
- `asr_extended_latency_gpu_29.csv`：逐筆延遲與 RTF
- `asr_extended_language_summary_gpu_29.csv`：國語、台語、國台混合、中英混合、客語（四縣）的 CER 與延遲摘要

若辨識已完成，只要重新彙整輸出，可執行：

```bat
python evaluation\scripts\evaluate_asr_extended.py --skip-transcription
```
