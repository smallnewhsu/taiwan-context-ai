# 語你傳心 v0.5 凍結獨立測試規範

## 目的

本測試集用來驗證語你傳心在未參與既有規則調整之案例上的泛化能力，避免以同一批案例同時進行規則開發與最終成效宣稱。

## 凍結程序

1. 執行評測前，先將 `test_cases_v0.5_holdout.json` 與本規範提交至Git。
2. 記錄提交編號與測試檔SHA-256。
3. 使用已凍結的程式版本與模型執行一次正式評測。
4. 看到評測結果後，不修改案例、必要詞、同義詞群組或禁止詞。
5. 自動失敗案例保留在正式結果中，以錯誤分析說明，不回頭調整成通過。
6. 若日後修正模型或安全規則，另建新版本並再次使用新的獨立測試集。

## 案例組成

共12筆，涵蓋家庭、朋友、職場、醫療、公共服務、台語及中英混合情境。案例特別檢查數字、日期、否定、不確定性、限制語氣、範圍、地點及禁止自行新增承諾。

## 執行方式

```bat
python tools\evaluate_rewrite.py --cases evaluation\rewrite\test_cases_v0.5_holdout.json --output-dir evaluation\rewrite\results\v0.5_holdout_qwen1.5b_gpu
```

## 雜湊紀錄

在Windows命令提示字元執行：

```bat
certutil -hashfile evaluation\rewrite\test_cases_v0.5_holdout.json SHA256
git rev-parse HEAD
```

將兩項輸出記錄在正式評測報告中，以證明測試案例在評測前已凍結。
