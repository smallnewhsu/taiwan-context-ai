\# Taiwan Tongues ASR CE Baseline Evaluation



\## 測試環境



\- Model: Taiwan-Tongues-ASR-CE-v1.0

\- Device: NVIDIA CUDA GPU

\- Compute Type: float16

\- Number of Audio Files: 5

\- Audio Format: M4A

\- Evaluation Metric: Character Error Rate（CER）



\## Baseline Results



\- Mean CER: 0.4288

\- Mean Accuracy: 57.12%

\- Substitution Errors: 24

\- Deletion Errors: 3

\- Insertion Errors: 3



\## Preliminary Findings



模型對純國語語句具有良好辨識能力，但在臺灣台語詞彙及國臺混合語句中，仍出現同音詞、近音詞與在地用語替換錯誤。



例如：



\- 「週末」辨識為「做夢」

\- 「轉去」辨識為「回來」

\- 「無閒」辨識為「忙」



後續將透過在地詞彙後處理、語境提示、使用者修正資料及模型微調進行改善。

