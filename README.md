# AI翻譯不了的台灣｜Taiwan Context AI

> 讓AI聽懂台灣人的話外之意。

Taiwan Context AI是一套在本機執行的多模態溝通與識讀系統，整合語音、文字與影像理解，並將人物關係、生活情境及使用者確認納入處理流程。

## 核心功能

|模組|功能|
|---|---|
|聲入其境|辨識國語、台語、客語、英語、越南語及混合語音，提供字面意思、可能意境、判斷依據與建議回應|
|語你傳心|依溝通對象調整表達方式，並檢查數字、否定、限制條件及新增資訊|
|視界有解|分析台灣生活圖片、擷取可見文字，並支援影像後接語音或文字追問|

一般使用者可使用三項核心功能並確認AI輸出；管理者負責回饋審查、資料授權、資料集發布、評測與服務狀態管理。

## 技術架構

```text
語音／文字／影像
        ↓
Taiwan Tongues ASR CE／Whisper／Gemma 3 4B
        ↓
Taiwan Context Engine
        ↓
語意安全檢查與使用者確認
        ↓
授權回饋 → 人工審查 → 去識別化資料集
```

- Taiwan Tongues ASR CE：國語、台語、客語、英語與混合語音辨識
- faster-whisper：越南語辨識
- Gemma 3 4B：文字改寫、語境分析、繁中翻譯、影像理解與多模態問答
- Taiwan Context Engine：關係判斷、語意保真、安全回退及人工確認

正式競賽版本使用Google Gemma 3 4B。歷史開發結果不列入正式成果。

## 評測摘要

|項目|結果|
|---|---:|
|ASR T22～T50共29筆，v3整體平均CER|36.80%|
|語你傳心36筆擴充集，自動安全通過率|100%|
|語你傳心12筆凍結獨立測試，自動首次通過率|83.33%|
|語你傳心12筆凍結獨立測試，人工複核原始輸出通過率|91.67%|
|語你傳心人工確認與修正後安全完成率|100%|
|視界有解12張圖片，安全確認判定準確率|100%|

「人工確認與修正後安全完成率100%」不等同於模型首次輸出準確率。完整評測資料與限制請見[`evaluation/`](evaluation/README.md)。

## 資料治理

```text
使用者修正 → 明確授權 → 待審回饋 → 人工審查 → 去識別化候選 → 版本化發布
```

未授權資料不會加入資料集；私人語音與圖片不提交至公開Git儲存庫。

## 快速啟動

需求：Windows 10/11、Python 3.10、Ollama、FFmpeg及NVIDIA CUDA GPU。

```bat
git clone --recurse-submodules https://github.com/smallnewhsu/taiwan-context-ai.git
cd /d taiwan-context-ai
ollama pull gemma3:4b
start_demo.bat
```

- 首頁：<http://127.0.0.1:8000/app/>
- 管理中心：<http://127.0.0.1:8000/app/admin/>
- API文件：<http://127.0.0.1:8000/docs>

操作流程請見[`docs/DEMO_RUNBOOK.md`](docs/DEMO_RUNBOOK.md)，系統設計請見[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 專案目錄

```text
backend/                  FastAPI、模型服務、安全層與資料治理
frontend/                 使用者介面、登入與管理中心
datasets/taiwan_context/  詞彙、回饋與資料集版本
evaluation/               ASR、文字改寫及視覺評測
services/asr/             Taiwan Tongues ASR CE子模組
tools/                    評測與資料匯出工具
docs/                     架構與Demo文件
```

## 已知限制

- 台語、國台混合與客語的語音辨識仍需更多說話者資料。
- 越南語獨立測試集尚待完成。
- 小型VLM的繁體中文OCR可能漏字或產生幻覺，必須保留人工確認。
- 目前人工評測者數量有限，尚未建立評分者間一致性。
- 本系統提供溝通輔助，不取代專業判斷或使用者決策。

## 授權

本專案自行開發的程式碼採Apache License 2.0。模型、資料與第三方元件仍適用各自條款，詳見[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
