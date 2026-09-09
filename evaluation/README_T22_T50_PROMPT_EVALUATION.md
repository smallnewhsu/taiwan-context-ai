# T22～T50 分語言提示優化 v2 評測

本評測不修改模型權重，也不覆蓋基準結果。它使用與專案目前
`services/asr/evaluate_asr.py` 相同的模型、文字正規化及 CER 計算方式，
僅針對台語、客語（四縣）與國台混合加入不含範例詞彙的短提示，
避免第一版詞彙清單造成補字與語意誘導。

## 安裝

將 ZIP 解壓縮到 `C:\taiwan-context-ai`，確認檔案位於：

`C:\taiwan-context-ai\evaluation\scripts\evaluate_asr_prompt_extended.py`

## 執行

```bat
cd /d C:\taiwan-context-ai
call services\asr\asr_api\Scripts\activate.bat
python evaluation\scripts\evaluate_asr_prompt_extended.py
```

執行前須保留基準結果：

`evaluation\asr_extended\asr_extended_results_gpu_29.json`

## 輸出

程式會在 `evaluation\asr_extended\` 產生：

- `asr_prompt_v2_results_gpu_29.json`
- `asr_prompt_v2_comparison_gpu_29.csv`
- `asr_prompt_v2_language_comparison_gpu_29.csv`
- `asr_prompt_v2_latency_gpu_29.csv`
- `T22_prompt_v2_asr.txt` 至 `T50_prompt_v2_asr.txt`

其中比較表的 `absolute_cer_change` 小於零代表改善；
`relative_improvement_rate` 大於零代表錯誤率下降。
