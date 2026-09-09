# 視界有解：多模態追問修正版

將本資料夾內下列檔案覆蓋到 `C:\taiwan-context-ai` 的相同位置：

- `backend\api.py`
- `frontend\vision\index.html`
- `frontend\vision\vision.js`
- `frontend\vision\vision-followup.css`

啟動 Ollama 與 FastAPI 後，以瀏覽器開啟：

`http://127.0.0.1:8000/app/vision/`

本版本支援：

1. 直接使用瀏覽器麥克風錄音提問。
2. 上傳既有問題音檔。
3. 直接輸入文字提問。
4. 三種方式共用前一輪影像分析內容與 Taiwan Context Engine 回答流程。

首次錄音時需允許瀏覽器使用麥克風。不要使用 `file:///` 方式直接開啟 HTML。
