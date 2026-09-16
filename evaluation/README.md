# Taiwan Context AI評測總覽

本目錄保存測試案例、標準答案、逐筆輸出、延遲與人工複核資料。正式成果與歷史開發結果分開呈現。

## 正式成果

|模組|版本|重點結果|
|---|---|---|
|聲入其境|T22～T50 v3|29筆；平均CER 36.80%|
|語你傳心|v0.6.2擴充集|36筆；自動安全通過率100%|
|語你傳心|v0.6.3凍結獨立測試|12筆；自動83.33%、人工原始輸出91.67%、修正後安全完成100%|
|視界有解|v0.1.1|12張；安全確認判定準確率100%|

## ASR評測

```bat
python evaluation\scripts\evaluate_asr_extended.py
python evaluation\scripts\evaluate_asr_prompt_extended.py
python evaluation\scripts\build_asr_hybrid_v3.py
```

T22～T50包含國語8筆、台語7筆、國台混合6筆、中英混合4筆及客語4筆。v3為同批資料上的探索性路由優化，整體平均CER由37.14%降至36.80%，不能視為獨立泛化證據。

T51～T80為後續台語、客語與越南語獨立評測規劃；尚未完成錄音與正式評測前，不列入成果數字。

## 語你傳心評測

先啟動後端，再執行：

```bat
python tools\evaluate_rewrite.py --cases evaluation\rewrite\test_cases_v0.4_expanded.json --api-url http://127.0.0.1:8000/expression/rewrite --output-dir evaluation\rewrite\results\v0.6.2_expanded_gemma3_4b_gpu
python tools\evaluate_rewrite.py --cases evaluation\rewrite\test_cases_v0.5_holdout.json --api-url http://127.0.0.1:8000/expression/rewrite --output-dir evaluation\rewrite\results\v0.6.3_holdout_gemma3_4b_gpu
```

凍結測試的測試檔SHA-256為`827b3ff448a0d22122acb0fa99c35668fd640ac388657a902e4f40c44e613631`。

## 視界有解評測

```bat
python tools\evaluate_vision.py
```

## 解讀原則

- CER越低代表字元錯誤越少。
- 自動規則通過率不是模型準確率。
- 人工複核與人工修正後安全完成必須分開報告。
- OCR嚴格召回率不能以人工觀感分數取代。
- 開發集成果不能取代凍結獨立測試。
- 歷史模型資料夾僅供版本追溯，不列入正式競賽成果。
