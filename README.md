# AI翻譯不了的台灣｜Taiwan Context AI

> 讓AI聽懂台灣人的話外之意。

以開源AI模型打造在地化、無礙且可由使用者確認的數位溝通與識讀系統。系統不只辨識語音、文字與圖片，也將人物關係、前後文及台灣生活情境納入理解流程。

## 為什麼需要這套系統？

「你食飽未？若無就轉來阮兜食啦。」字面上是在詢問吃飯，也可能包含長輩的關心與邀請。一般翻譯只處理字詞，Taiwan Context AI則把話語放回關係與生活情境中；證據不足時要求使用者確認，不替使用者武斷下結論。

## 三個核心模組

|模組|輸入|核心功能|輸出|
|-|-|-|-|
|**聲入其境**|即時錄音或音檔|依語言選擇Taiwan Tongues ASR CE或multilingual Whisper，辨識國語、台語、客語、英語、越南語及混合語音，再結合人物關係與前後文|辨識話語、繁中字面意思、可能意境、判斷依據、建議回應|
|**語你傳心**|語音或文字|Qwen2.5 1.5B依溝通對象調整表達，並執行語意保真檢查|直接表達、安心表達、台語表達及安全警告|
|**視界有解**|圖片、語音或文字|Gemma 3分析生活影像，再以語音或文字接續詢問|物件、可見文字、生活情境、可能意圖及追問回答|

所有AI輸出皆可由使用者確認與修改。系統不會把低信心辨識、OCR候選或文化推測包裝成確定事實。

## 競賽加分成果

### 1\. 核心模型實質介接與在地優化

後端直接介接數產署開源的 [Taiwan Tongues ASR CE](https://github.com/adi-gov-tw/Taiwan-Tongues-ASR-CE)，不是以預錄結果模擬。專案針對五類生活語音建立29筆擴充評測，並設計語言提示與條件式推論路由。

|語言|案例數|基準Macro CER|v3路由Macro CER|
|-|-:|-:|-:|
|國語|8|3.24%|3.24%|
|台語|7|74.97%|74.97%|
|國台混合|6|38.77%|38.41%|
|中英混合|4|7.40%|7.40%|
|客語|4|66.08%|64.16%|
|**整體**|**29**|**37.14%**|**36.80%**|

v3依語言類型選擇基準或Prompt v2路徑，使整體平均CER由37.14%降至36.80%。本階段屬推論流程優化，未修改模型權重；後續將用獨立測試集驗證泛化能力。本批客語音檔以四縣腔為主。

### 多語言辨識與繁中字面翻譯

「聲入其境」已加入可明確選擇的語言路由。既有國語、台語、客語、英語及混合語音維持Taiwan Tongues ASR CE路徑；越南語則使用`faster-whisper`的multilingual Whisper路徑，避免影響原有語言的辨識流程。

|選擇語言|辨識路徑|字面意思處理|
|-|-|-|
|國語、台語、客語、英語及混合語音|Taiwan Tongues ASR CE|Taiwan Context Engine產生繁中字面意思|
|越南語|faster-whisper large-v3-turbo|先保留越南語逐字稿，再由Qwen2.5 1.5B翻譯成繁體中文|

若主要語境分析未產生中文、輸出空白或僅照抄越南語，後端會自動執行一次獨立的「越南語→繁體中文」翻譯重試；若仍失敗，介面會要求使用者確認逐字稿，不以原文冒充字面意思。此功能已完成程式流程驗證，但越南語獨立測試集仍在擴充，因此不宣稱已達正式服務品質。

多語評測規格與後續驗證方式見[T51～T80台語、客語及越南語獨立評測規格](evaluation/README_T51_T80_MULTILINGUAL.md)。

### 2\. 影像後接語音問答

使用者可先上傳紙條、公告、店面標示或生活物件，由Gemma 3產生影像觀察與OCR候選，再使用瀏覽器麥克風、既有問題音檔或文字追問，例如「她是在生氣嗎？」或「我應該怎麼回覆？」。後端結合前一輪影像內容、Taiwan Tongues ASR CE辨識結果及Taiwan Context Engine產生語境回答。

```text
生活圖片 → Gemma 3影像理解 → 暫存影像語境
                               ↓
錄音／音檔 → Taiwan Tongues ASR CE → Taiwan Context Engine → 可確認回答
```

### 3\. 可審查、可授權、可發布的資料循環

```text
使用者修正 → 明確授權 → 待審回饋 → 人工審查 → 去識別化候選 → 版本化發布
```

系統記錄原始輸出、人工修正、同意狀態及音檔SHA-256。未授權資料不會成為資料集候選；只有人工核准且去識別化的內容才能進入Taiwan Context Dataset正式版本。

## 安全設計

* **語音：** 候選不一致或人物關係證據不足時要求確認。
* **文字：** 保護數字、日期、否定、限制語氣、台語詞彙及省略資訊；未通過語意保真檢查時自動回退。
* **影像：** 分開呈現直接觀察、OCR候選、情境推測與無法確認資訊。
* **隱私：** 原始語音與圖片不納入Git，私人影像只在本機暫時處理。
* **資料：** 使用者授權、人工審查與正式發布版本分開管理。

## 實測摘要

### 語你傳心：輕量化與36筆擴充評測

|指標|Qwen2.5 3B歷史基準|Qwen2.5 1.5B原12筆|Qwen2.5 1.5B v0.4.1擴充集|
|-|-:|-:|-:|
|測試案例|12|12|36|
|自動安全通過率|100%|100%|100%|
|LLM候選採用率|41.67%|41.67%|72.22%|
|安全回退率|58.33%|58.33%|27.78%|
|穩態平均延遲|1.155秒|0.673秒|0.609秒|
|穩態P95|1.356秒|0.852秒|0.707秒|
|人工整體平均|4.92/5|尚未另行評分|4.98/5|
|人工可接受率|—|—|100%（36/36）|

v0.4.1擴充集涵蓋家庭、朋友、職場、醫療照護、公共服務，以及台語與中英混合情境。36筆皆通過自動安全檢查；26筆直接採用LLM候選，另外10筆因新增資訊、事實變動、否定或在地詞彙風險而觸發安全回退。10筆回退結果的人工平均仍達4.93/5且全部可接受，顯示安全層能在保留可用性的同時攔截高風險候選。

完整案例、原始輸出、人工評閱與限制說明見[語你傳心 v0.4.1擴充評測報告](evaluation/rewrite/results/v0.4.1_expanded_qwen1.5b_gpu/rewrite_v0.4.1_final_report.md)。

### 視界有解：辨識風險並交還決定權

12張生活圖片皆完成端到端處理，安全確認判定準確率100%，人工有效評分平均4.40/5。小型VLM的繁體中文OCR仍有限制，因此低信心OCR與文化推論均標示為待確認內容。完整結果見[視覺評測報告](evaluation/vision/results/v0.1.1_gpu/vision_v0.1.1_final_report.md)。

## 系統架構

```text
錄音／文字／圖片 → FastAPI本機服務
        ↓
Taiwan Tongues ASR CE／Qwen2.5 1.5B／Gemma 3 4B
        ↓
Taiwan Context Engine → 安全檢查、人工確認與多模態追問
        ↓
授權修正 → 人工審查 → Taiwan Context Dataset版本
```

詳細資料流與安全邊界見[系統架構文件](docs/ARCHITECTURE.md)。

## 快速啟動（Windows）

需求：Windows 10/11、Python 3.10專案環境、Ollama、FFmpeg／FFprobe及NVIDIA CUDA GPU（建議6GB以上）。Windows VAD使用`webrtcvad-wheels==2.0.14`。

```bat
git clone --recurse-submodules https://github.com/smallnewhsu/taiwan-context-ai.git
cd /d taiwan-context-ai
ollama pull qwen2.5:1.5b
ollama pull gemma3:4b
services\\asr\\asr\_api\\Scripts\\python.exe -c "from huggingface\_hub import snapshot\_download; snapshot\_download('mobiuslabsgmbh/faster-whisper-large-v3-turbo')"
start\_demo.bat
```

* 首頁：[http://127.0.0.1:8000/app/](http://127.0.0.1:8000/app/)
* 聲入其境：[http://127.0.0.1:8000/app/speech/](http://127.0.0.1:8000/app/speech/)
* 語你傳心：[http://127.0.0.1:8000/app/rewrite/](http://127.0.0.1:8000/app/rewrite/)
* 視界有解：[http://127.0.0.1:8000/app/vision/](http://127.0.0.1:8000/app/vision/)
* 管理中心：[http://127.0.0.1:8000/app/admin/](http://127.0.0.1:8000/app/admin/)
* API文件：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

完整流程見[離線Demo操作手冊](docs/DEMO_RUNBOOK.md)。所有評測程式、標準逐字稿與結果均保留在`evaluation/`，請由[評測總覽](evaluation/README.md)進入。

## 專案目錄

```text
backend/                 FastAPI、AI服務、安全層與資料治理
frontend/                使用者介面、登入與管理中心
datasets/taiwan\_context/ 詞彙、授權回饋與資料集版本
evaluation/              ASR、改寫及視覺評測
services/asr/            Taiwan Tongues ASR CE子模組
tools/                   評測與資料匯出工具
docs/                    架構與Demo文件
```

## 凍結獨立泛化評測

為避免以同一批案例同時開發規則並宣稱成效，本專案另建立12筆「語你傳心」凍結獨立測試集。測試案例在正式推論前提交至Git，並記錄測試檔SHA-256；結果產生後不修改案例或判定條件。

|指標|v0.5凍結獨立測試|
|-|-:|
|完成案例|12/12|
|API錯誤|0|
|嚴格字詞自動通過率|91.67%（11/12）|
|人工語意複核通過率|100%（12/12）|
|忠實度／對象適切性／自然度|5.00／5.00／4.92（滿分5）|
|LLM候選直接採用率|41.67%（5/12）|
|安全回退率|58.33%（7/12）|
|穩態平均推論時間|0.650秒|
|穩態P95|0.873秒|

唯一自動未通過案例為模型將「誤點」改寫為等義的「晚點」，因凍結條件採嚴格必要詞比對而判定失敗。專案保留原始91.67%結果，不於評測後修改條件；經12筆逐案人工確認，語意複核通過率為100%。

* 凍結commit：`1056cc3267d08a0ffb9af0c92e493d2ff8de0442`
* 測試檔SHA-256：`827b3ff448a0d22122acb0fa99c35668fd640ac388657a902e4f40c44e613631`



## 已知限制

* 目前屬先導驗證，部分評測集規模仍小。
* 台語、國台混合及客語仍有較高CER，需要更多說話者與獨立測試資料。
* 小型VLM的繁體中文OCR可能漏字或產生幻覺，必須保留人工確認。
* 6GB GPU不適合同時常駐所有模型，Demo採依序載入。
* 正式研究仍需增加評測者並檢驗評分者間一致性。

## 模型與授權

* Taiwan Tongues ASR CE：TRAIL v0.1，含linking exception
* Qwen2.5 1.5B Instruct：Apache License 2.0
* Gemma 3 4B：Google Gemma Terms
* OpenAI Whisper／faster-whisper large-v3-turbo：依上游授權條款，用於越南語辨識

本專案自行開發的程式碼採Apache License 2.0；模型、資料與第三方套件仍依各自授權條款使用。詳見[第三方授權說明](THIRD_PARTY_NOTICES.md)。

