# Taiwan Tongues ASR CE先導基準評測

## 測試環境

- Model：Taiwan-Tongues-ASR-CE-v1.0
- Device：NVIDIA CUDA GPU
- Compute Type：float16
- Number of Audio Files：5
- Audio Format：M4A
- Evaluation Metric：Character Error Rate（CER）

## 基準結果

- Mean CER：0.4288
- Mean Accuracy：57.12%
- Substitution Errors：24
- Deletion Errors：3
- Insertion Errors：3

## 初步發現

模型對純國語語句具有較佳辨識能力，但在台語詞彙及國台混合語句中，仍出現同音詞、近音詞與在地用語替換錯誤，例如：

- 「週末」辨識為「做夢」
- 「轉去」辨識為「回來」
- 「無閒」辨識為「忙」

後續已透過語言提示、語言條件路由、在地詞彙保護與人工修正記憶進行推論流程優化。目前未宣稱完成模型權重微調；擴充結果請見[評測總覽](../README.md)。
