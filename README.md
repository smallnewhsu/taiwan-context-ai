# AI翻譯不了的台灣｜Taiwan Context AI

以開源AI模型打造在地化、無障礙且可被使用者確認的數位溝通與識讀系統。系統不只進行語音、文字或圖片辨識，也將臺灣生活中的語言混用、家庭表達與文化情境納入理解流程。

## 三個核心模組

### 聲入其境

使用 Taiwan Tongues ASR CE 執行本機GPU語音辨識，透過原始與臺灣語境提示雙路徑推論，再由 Taiwan Context Engine 選擇候選結果。當候選內容不一致或信心不足時，系統要求使用者確認，不直接覆寫原意。

### 語你傳心

使用本機 Ollama 與 Qwen2.5 1.5B Instruct 進行表達改寫，提供長輩、家人、朋友與正式情境的不同語氣。安全檢查會保護數字、日期、否定、限制語氣、臺灣台語詞彙與省略主詞，候選未通過語意保真時自動改用保守式回退結果。

### 視界有解

使用本機 Ollama 與 Gemma 3 4B 分析臺灣生活圖片。系統將輸出拆分為直接觀察、OCR候選、情境推測與無法確認資訊，並對低可信度文字、情境推論及可能幻覺提出警告。上傳圖片只在本機暫時處理，不永久保存。

## 系統架構

```text
語音／文字／圖片
        ↓
FastAPI Backend
        ↓
ASR、LLM、VLM開源模型
        ↓
Taiwan Context Engine
        ↓
安全檢查與人工確認
        ↓
去識別授權修正 → 人工審查 → Taiwan Context Dataset
```

## 快速啟動（Windows）

需求：

- Windows 10或11
- Python 3.10專案虛擬環境
- Ollama
- FFmpeg與FFprobe
- NVIDIA CUDA GPU（建議6GB以上顯示記憶體）

安裝模型：

```bat
ollama pull qwen2.5:1.5b
ollama pull gemma3:4b
```

啟動：

```bat
start_demo.bat
```

開啟：

- 系統首頁：<http://127.0.0.1:8000/app/>
- 語你傳心：<http://127.0.0.1:8000/app/rewrite/>
- 視界有解：<http://127.0.0.1:8000/app/vision/>
- API文件：<http://127.0.0.1:8000/docs>

## API

| 方法 | 路徑 | 功能 |
|---|---|---|
| GET | `/health` | 後端健康檢查 |
| GET | `/llm/health` | 本機文字模型檢查 |
| GET | `/vision/health` | 本機視覺模型檢查 |
| POST | `/speech/interpret` | 雙路徑語音辨識與情境選擇 |
| POST | `/context/select` | 候選文字選擇 |
| POST | `/expression/rewrite` | 安全語意改寫 |
| POST | `/vision/analyze` | 圖片理解與幻覺風險提示 |
| POST | `/feedback` | 儲存具同意狀態的使用者修正 |
| GET | `/feedback/pending` | 取得待審修正 |
| POST | `/feedback/{id}/review` | 人工核准或拒絕修正 |

## 評測結果

### 聲入其境

- 10段先導語音平均CER：0.5101
- GPU平均推論時間：0.707秒
- 排除暖機後平均：0.668秒
- 排除暖機後P95：0.789秒
- Prompt平均CER：0.4590
- 雙路徑獨立驗證：路由CER 0.5390、Oracle CER 0.5319

### 聲入其境 T22～T50擴充評測

擴充評測收錄29段多語生活語音及其標準逐字稿，包含國語8段、臺灣台語7段、國台混合6段、中英混合4段，以及客語腔4段。

| 語言類型 | 案例數 | 基準Macro CER | 語言條件路由v3 |
|---|---:|---:|---:|
| 國語 | 8 | 3.24% | 3.24% |
| 臺灣台語 | 7 | 74.97% | 74.97% |
| 國台混合 | 6 | 38.77% | 38.41% |
| 中英混合 | 4 | 7.40% | 7.40% |
| 客語 | 4 | 66.08% | 64.16% |
| **整體** | **29** | **37.14%** | **36.80%** |

- 29/29案例完成標準逐字稿比對，基準平均正確率為62.86%。
- 詞彙提示v1平均CER為37.59%，因提示誘導造成錯誤補字，未納入正式路由。
- 不含詞彙範例的短提示v2平均CER為37.09%；排除暖機後平均推論時間為0.649秒，P95為0.748秒。
- 語言條件路由v3讓國語、臺灣台語與中英混合維持基準策略，國台混合與客語四縣腔使用v2短提示，平均CER為36.80%，相較基準相對改善0.91%。
- v3路由規則由同一批擴充案例分析而得，目前屬探索性推論策略優化，仍須使用獨立測試集驗證泛化能力；本專案未宣稱進行模型權重微調。

評測文件與可重現結果：

- [T22～T50基準評測說明](evaluation/README_T22_T50_EVALUATION.md)
- [T22～T50提示評測說明](evaluation/README_T22_T50_PROMPT_EVALUATION.md)
- [29份標準逐字稿與評測輸出](evaluation/asr_extended/)
- [語言條件路由v3詳細結果](evaluation/asr_extended/asr_hybrid_v3_results_gpu_29.json)

### 語你傳心 v0.2.3（Qwen2.5 3B 歷史基準）

- 12/12案例完成
- 自動安全檢查通過率：100%
- LLM候選直接採用率：41.67%
- 安全回退率：58.33%
- 穩態平均時間：1.155秒
- 穩態P95：1.356秒
- 人工整體平均：4.92/5

目前預設文字模型已切換為 Qwen2.5 1.5B Instruct。為避免把不同模型的結果混為一談，1.5B 評測請輸出至 `evaluation/rewrite/results/v0.3_qwen1.5b_gpu/`，完成後再新增其實測指標。

### 視界有解 v0.1.1

- 12/12案例完成，API錯誤0次
- 物件召回率：60.78%
- OCR完全文字召回率：40.00%
- 情境關鍵概念召回率：58.33%
- 原始輸出無幻覺率：16.67%
- 安全確認判定準確率：100%
- 穩態平均時間：10.643秒
- 穩態P95：23.167秒
- 人工有效評分平均：4.40/5

## 安全與隱私原則

- 不將低信心模型輸出包裝成確定事實。
- 數字、日期、否定、限制語氣與在地詞彙優先保真。
- 原始語音與圖片不納入Git版本庫。
- 使用者修正只有在明確同意後才能進入待審資料。
- 所有資料在進入公開資料集前必須經人工審查與去識別。
- VLM輸出僅供輔助；OCR候選與文化推論均需人工確認。

## 專案目錄

```text
backend/                 FastAPI與各AI服務
frontend/                三模組網頁介面與人工審查台
datasets/taiwan_context/ 語境詞彙、回饋流程與資料集版本
evaluation/              ASR、改寫及視覺評測結果
services/asr/            Taiwan Tongues ASR子模組
tools/                   評測與資料匯出工具
docs/                    技術與Demo文件
```

## 已知限制

- 現階段評測集規模較小，屬先導與可行性驗證。
- 臺灣台語及國臺混合語音仍有較高CER。
- 視覺模型的繁體中文OCR與具體文字生成仍有明顯幻覺風險。
- 6GB GPU需避免ASR、文字模型與視覺模型同時常駐。
- 正式研究需增加說話者、圖片來源與獨立測試集，並進行多位評閱者一致性分析。

## 模型與授權

- Taiwan Tongues ASR CE（TRAIL v0.1；含 linking exception）
- Qwen2.5 1.5B Instruct（Apache License 2.0）
- Gemma 3 4B（Google Gemma Terms）

本專案自行開發的程式碼採 Apache License 2.0；模型、資料與第三方套件仍依各自授權條款使用。詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
