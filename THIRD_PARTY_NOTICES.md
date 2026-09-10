# Third-party models and components

Taiwan Context AI 自行開發的程式碼依專案根目錄 `LICENSE`（Apache License 2.0）授權。下列模型及外部元件不因此改採 Apache-2.0，使用者仍須遵守其各自條款。

| 元件 | 本專案用途 | 上游授權／條款 | 官方來源 |
|---|---|---|---|
| Taiwan Tongues ASR CE | 多語語音辨識 | TRAIL v0.1；其 linking exception 允許連結應用採其他授權 | https://github.com/adi-gov-tw/Taiwan-Tongues-ASR-CE |
| OpenAI Whisper／faster-whisper large-v3-turbo | 越南語語音辨識 | Whisper程式碼與轉換模型依各自上游條款 | https://github.com/SYSTRAN/faster-whisper |
| Qwen2.5 1.5B Instruct | 文字語境分析、安全改寫與多模態問答 | Apache License 2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| Gemma 3 4B | 影像理解 | Google Gemma Terms | https://ai.google.dev/gemma/terms |
| Ollama | 本機模型執行環境 | 依 Ollama 上游專案授權 | https://github.com/ollama/ollama |

## 說明

- 本專案不重新授權第三方模型權重。
- 下載、使用、修改或散布模型前，請直接查閱上游最新條款。
- `語你傳心 v0.2.3` 的既有數據是 Qwen2.5 3B 歷史基準，不代表 Qwen2.5 1.5B 的表現。
