# T51～T80多語言錄音套件

## 案例配置

- T51～T60：台語10筆
- T61～T70：客語10筆（本套件以四縣腔用詞設計，系統分類仍顯示「客語」）
- T71～T80：越南語10筆

## 錄音前必要程序

1. 請實際朗讀者逐句檢查文字是否符合自己的自然說法。
2. 台語與客語若因腔調調整用字，必須同步修改對應的`Txx_reference.txt`，再開始錄音。
3. 越南語逐字稿須由熟悉越南語者確認聲調符號、否定、時間、數字及醫療詞彙。
4. reference必須記錄「實際說出的原語言」，不可先翻成國語。
5. 錄音與逐字稿全數確認後，才計算SHA-256並提交Git凍結。

## 錄音規格

- 格式：M4A（AAC-LC）
- 建議取樣率：44.1 kHz或48 kHz
- 單聲道或雙聲道皆可，整批設定應一致
- 每檔只讀一個案例，句首與句尾各保留約0.5秒安靜
- 正常語速、自然口氣，不刻意逐字停頓
- 檔名必須為`T51.m4a`至`T80.m4a`
- 不加入背景音樂、變聲或降噪特效

## 檔案放置方式

將錄好的M4A與本套件的reference放在同一目錄：

```text
evaluation/asr_multilingual/T51.m4a
evaluation/asr_multilingual/T51_reference.txt
...
evaluation/asr_multilingual/T80.m4a
evaluation/asr_multilingual/T80_reference.txt
evaluation/asr_multilingual/metadata.csv
```

## 重要限制

本批案例完成錄音與人工核稿後才是獨立測試集。看到辨識結果後，不得回頭修改reference或路由規則，再以同一批資料宣稱泛化改善；若必須修正，應建立新版本並完整保留原始結果。

