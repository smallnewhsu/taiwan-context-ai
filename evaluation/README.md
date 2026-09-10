# Taiwan Context AI評測總覽

本目錄保存可重現的測試程式、標準答案、逐筆輸出與摘要。歷史版本不覆蓋，避免不同模型或規則的結果混淆。

## 評測項目

| 模組 | 評測內容 | 主要文件／結果 |
|---|---|---|
| 聲入其境 | 先導ASR基準 | [asr_baseline](asr_baseline/) |
| 聲入其境 | T22～T50共29筆多語基準 | [基準評測說明](README_T22_T50_EVALUATION.md) |
| 聲入其境 | Prompt v2與v3語言條件路由 | [提示評測說明](README_T22_T50_PROMPT_EVALUATION.md) |
| 語你傳心 | 安全改寫、自動檢查、延遲與人工評分 | [rewrite](rewrite/) |
| 視界有解 | 物件、OCR、情境、幻覺安全與延遲 | [vision](vision/) |

## ASR擴充評測

```bat
call services\asr\asr_api\Scripts\activate.bat
python evaluation\scripts\evaluate_asr_extended.py
python evaluation\scripts\evaluate_asr_prompt_extended.py
python evaluation\scripts\build_asr_hybrid_v3.py
```

分類名稱統一使用國語、台語、國台混合、中英混合及客語。本批客語測試音檔以四縣腔為主，腔別只在方法說明中註明，不作為介面分類名稱。

## 語你傳心1.5B評測

先啟動後端，再執行：

```bat
python tools\evaluate_rewrite.py --output-dir evaluation\rewrite\results\v0.3_qwen1.5b_gpu
```

## 視界有解評測

```bat
python tools\evaluate_vision.py
```

## 解讀原則

- CER越低代表字元錯誤越少。
- 自動安全通過不等於自然度良好，仍需人工評閱。
- OCR完全命中率是嚴格指標，不能以人工整體觀感取代。
- 路由規則若由同一批案例建立，結果應標示為探索性成果並另建獨立測試集。
- 不得把3B人工評分直接套用到1.5B輸出。
