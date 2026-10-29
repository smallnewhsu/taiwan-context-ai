# AI翻譯不了的台灣｜Taiwan Context AI

> 讓AI不只辨識文字，也知道何時需要請人確認。

Taiwan Context AI是一套面向台灣生活情境的多模態溝通與識讀系統，整合語音、文字與影像模型，並以Taiwan Context Engine處理語境、安全檢查、低信心提示及人工確認。

本專案以數位發展部數位產業署開源的Taiwan Tongues ASR CE為語音核心，完成台語LoRA在地化調適、越南語獨立路由，以及文字與影像的語意安全流程。

## 三項核心功能

|模組|使用者功能|安全設計|
|---|---|---|
|聲入其境|辨識國語、台語、客語、英語、越南語及混合語音，提供字面意思與可能語境|雙候選比較、低信心提示、使用者確認與人工修正|
|語你傳心|依溝通對象調整表達方式|檢查數字、否定、限制條件、事實變動及新增資訊|
|視界有解|分析生活圖片、擷取可見文字，支援影像後接文字或語音追問|區分畫面事實與推論，資訊不足時保留不確定性|

一般使用者可操作三項功能並確認AI輸出；管理者負責回饋審查、資料授權、資料集發布、評測與服務狀態管理。

## 技術流程

```text
語音／文字／影像
        ↓
Taiwan Tongues ASR CE／Whisper Turbo／Gemma 3 4B
        ↓
Taiwan Context Engine
        ↓
語意安全檢查與使用者確認
        ↓
明確授權 → 人工審查 → 去識別化資料集
```

|元件|用途|
|---|---|
|Taiwan Tongues ASR CE|國語、台語、客語、英語與混合語音辨識|
|台語LoRA＋CTranslate2 Float16|生活台語在地化辨識與GPU部署|
|Whisper Turbo|越南語辨識|
|Gemma 3 4B|文字改寫、語境分析、繁中翻譯、影像理解與問答|
|Taiwan Context Engine|候選選擇、語意保真、安全回退與人工確認|

核心模型均揭露來源與授權，未使用中國機構、企業或研究單位開發或維護的模型作為核心模型；第三方授權詳見[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 開源模型在地化擴充

|比較項目|原始Taiwan Tongues ASR CE|Taiwan Context AI|
|---|---|---|
|台語辨識|使用開源基礎模型|以官方開源語料進行LoRA調適，並轉換為CT2 Float16部署|
|越南語|未提供本專案可直接使用的越南語推論路徑|新增Whisper Turbo越南語路由與繁體中文字面翻譯|
|語言選擇|依原服務設定|可明確選擇國語、台語、客語、英語、越南語或自動判斷|
|輸出內容|語音逐字稿|加入字面意思、人物關係、可能語境、判斷依據及建議回應|
|風險處理|輸出辨識候選|比較無提示與語言提示候選，低信心時要求使用者確認|
|資料治理|依上游專案規範|加入明確授權、人工審查、去識別化與版本化發布流程|

## 主要評測結果

CER越低越好；自動通過率不等同於所有內容皆由模型首次正確產生，安全回退與人工確認另行揭露。

### 聲入其境

|評測|結果|
|---|---:|
|T51–T80多語測試，30筆整體Micro CER|34.83%|
|越南語10筆Micro CER|10.54%|
|台語LoRA獨立凍結測試，基礎模型Micro CER|71.93%|
|台語LoRA獨立凍結測試，調適後Micro CER|65.61%|
|台語LoRA相對改善|8.78%|
|CT2 Float16三次測試|共60筆，API錯誤0|
|CT2雙路徑三次平均時間|1.560秒|

台語最終測試集T81–T100共20筆、2位說話者，在v3訓練前完成凍結，與訓練逐字稿完全重複數為0。三次重複測試的CER與錯誤字數一致。完整方法、失敗案例及限制見[`evaluation/asr_finetune/TAIGI_LORA_FINAL_REPORT.md`](evaluation/asr_finetune/TAIGI_LORA_FINAL_REPORT.md)。

### 語你傳心

|評測|結果|
|---|---:|
|36筆擴充測試，自動安全通過率|100%|
|12筆凍結獨立測試，自動首次通過率|83.33%|
|12筆凍結測試，人工確認與修正後安全完成率|100%|

### 視界有解

|評測|結果|
|---|---:|
|12張圖片，自動通過率|91.67%|
|答案關鍵詞召回率|96.43%|
|無幻覺率|100%|
|不確定性判斷準確率|100%|

完整評測資料、測試案例與版本限制見[`evaluation/`](evaluation/README.md)。

## 語言條件路由

|使用者選擇|正式路由|
|---|---|
|台語|Taiwan Tongues ASR CE＋台語LoRA（CT2 Float16）|
|客語|Taiwan Tongues ASR CE基礎模型|
|國語、英語及混合語音|Taiwan Tongues ASR CE基礎模型|
|越南語|Whisper Turbo|

客語LoRA實驗未通過部署門檻，因此正式系統維持基礎模型與人工確認機制。專案保留改善與退步結果，避免只呈現成功實驗。

## 快速啟動

需求：Windows 10/11、Python 3.10、Ollama、FFmpeg及NVIDIA CUDA GPU。

```bat
git clone --recurse-submodules https://github.com/smallnewhsu/taiwan-context-ai.git
cd /d taiwan-context-ai
ollama pull gemma3:4b
start_demo.bat
```

台語CT2模型不直接存入Git。啟動前須將模型放在本機，並設定：

```bat
set "TAIGI_ASR_MODEL_PATH=C:\taiwan-context-training\models\taigi_lora_v3_ct2_float16"
```

模型目錄至少應包含`model.bin`、`config.json`、`tokenizer.json`、`vocabulary.json`、`preprocessor_config.json`及`generation_config.json`。

- 首頁：<http://127.0.0.1:8000/app/>
- 管理中心：<http://127.0.0.1:8000/app/admin/>
- API文件：<http://127.0.0.1:8000/docs>

操作流程請見[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md)，系統設計請見[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 資料治理

```text
使用者修正 → 明確授權 → 待審回饋 → 人工審查 → 去識別化候選 → 版本化發布
```

未授權資料不會加入資料集；私人語音與圖片不提交至公開Git儲存庫。凍結測試集以manifest與雜湊保存版本證據，重新錄製的診斷音檔不取代原始正式測試案例。

## 專案目錄

```text
backend/                  FastAPI、模型服務、安全層與資料治理
frontend/                 使用者介面、登入與管理中心
datasets/taiwan_context/  詞彙、回饋與資料集版本
evaluation/               ASR、文字改寫及視覺評測
services/asr/             Taiwan Tongues ASR CE子模組
tools/                    資料準備、訓練、轉換與評測工具
docs/                     架構與Demo文件
```

## 已知限制

- 台語LoRA在獨立測試的Micro CER為65.61%，仍需增加說話者、口音、設備與環境噪音資料。
- 客語LoRA未通過部署門檻；後續須先改善目標書寫形式與逐字稿品質。
- 小型VLM的繁體中文OCR仍可能漏字，資訊不足時必須保留人工確認。
- 目前獨立測試集與人工評測規模有限，結果不代表所有真實場域。
- 首次載入或切換大型語音模型會產生冷啟動延遲。
- 本系統提供溝通輔助，不取代專業判斷或使用者決策。

## 授權

本專案自行開發的程式碼採Apache License 2.0。模型、資料與第三方元件仍適用各自條款，詳見[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
