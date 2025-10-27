# HW3 — Spam Classification Demo (Streamlit)

此專案是一個簡單的垃圾簡訊（SMS/Email）分類演示與實驗環境，包含資料前處理步驟、基線訓練程式、以及基於 Streamlit 的互動式視覺化與即時預測介面。

README 內容整理如下：專案概述、資料來源與處理、如何在本地建立環境與執行訓練／Streamlit、主要檔案說明、以及如何進行擴充或除錯。

---

## 專案概述

這個作業 (HW3) 的目標是建置一個端到端的垃圾訊息分類 demo：
- 提供資料預處理步驟（datasets/processed 中有多個中間步驟檔案），
- 訓練簡單的基線分類器（`ml/spam_classification/train.py`），
- 並透過 Streamlit (`ml/spam_classification/app.py`) 提供視覺化、模型評估與單筆/批次預測介面。

目標使用者：想要快速查看資料、檢視模型效能、或以簡單介面測試自訂文字範例的使用者（教育或展示用途）。

---

## 主要功能

- 資料預處理流水（scripts、datasets/processed/steps 內保存各步驟結果）
- 訓練基線模型（Linear SVM / SVC 或類似 pipeline）並輸出模型檔與評估報告到 `ml/spam_classification/artifacts/`
- Streamlit 互動應用
	- 資料總覽（label 分布、訊息長度分布、top tokens）
	- 顯示預先計算的評估結果（confusion matrix、ROC、PR、threshold sweep）
	- 單筆文字即時預測（讀取 `.joblib` 模型）與批次預測與下載
	- 更友善的 UI（自訂 CSS、badge 顯示 spam/ham、Streamlit 主題設定）

---

## 資料檔案 (簡要)

- `datasets/sms_spam_no_header.csv`：原始資料（未處理）
- `datasets/processed/sms_spam_clean.csv`：最終清理後版本（可供 demo 使用）
- `datasets/processed/steps/`：中間每個預處理步驟的輸出（lower、no_urls、printable、no_punct、norm 等），方便追蹤清理流程

如果你要替換資料，請確保 CSV 有一個文字欄位（預設 `text_clean` 或 `text`）和一個 label 欄（預設 `label`，值可能是 `spam`/`ham` 或 1/0）。

---

## 目錄結構（摘要）

```
HW3/
├─ datasets/                # 原始與處理後的資料
├─ ml/spam_classification/  # Model training + Streamlit demo
│  ├─ app.py                # Streamlit 應用（UI 與預測）
│  ├─ train.py              # 訓練 script（輸出到 artifacts/）
│  ├─ artifacts/            # 模型、評估報告等輸出
│  └─ requirements.txt      # demo / training 需要的套件
├─ scripts/                 # 其他資料處理工具
└─ README.md                # 本檔
```

---

## 快速開始（在本地）

建議在 Linux/macOS 上使用虛擬環境。以下步驟假設你在 repo 根目錄。

1) 建立並啟用虛擬環境，安裝依賴

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/spam_classification/requirements.txt
```

2) 若要訓練基線模型（可選）

```bash
# 會將 artifacts 寫到 ml/spam_classification/artifacts/
python ml/spam_classification/train.py
```

訓練完會在 `ml/spam_classification/artifacts/` 看到 `svm_baseline.joblib` 與 `eval_baseline.json` 等。

3) 啟動 Streamlit 應用

```bash
streamlit run ml/spam_classification/app.py
```

預設 Streamlit 會載入 `datasets/processed/sms_spam_clean.csv`（若你勾選「使用 repo 內的 processed dataset」），或你也可以上傳自訂 CSV 檔進行批次預測。

當前 repo 已加入 `ml/spam_classification/.streamlit/config.toml` 以統一主題顏色（primary、background、text 等），並在 `app.py` 中加入額外 CSS 來改善按鈕、卡片與 badge 的呈現。

---

## Streamlit UI 的說明（已做的改善）

- 統一色票：使用 `PALETTE` 常數搭配 `.streamlit/config.toml` 設定主題。
- 可視化改進：將 top-metrics 用 `st.metric` 呈現、使用 CSS badge 顯示預測結果（紅=spam，綠=ham）、表格與卡片改為更現代的樣式。
- 容錯處理：若模型不存在或格式不符，會顯示友善訊息並保留操作流程。

如果你希望更接近某個特定 demo（例如更互動的圖表），我可以把 matplotlib 換成 Plotly，或把 token 列表改成分頁/折疊顯示。

---

## 開發者注意 / 常見問題

- 若要將 Streamlit 部署到遠端（例如 Streamlit Cloud、Heroku 等），請注意把 `artifacts/` 中的 model 與 `example_messages.csv` 一起上傳或改成從外部儲存載入。
- 如果在啟動時顏色或 CSS 看起來沒套用，嘗試清除瀏覽器快取或重新啟動 Streamlit server（`Ctrl+C` 停止，重新 run）。

---

## 如何貢獻

任何改善都是歡迎的：
- 改進 UI（Plotly 圖表、互動表格）
- 加入更多預處理步驟或用不同模型比較（例如 LogisticRegression、RandomForest、Transformer-based）
- 把 demo 改為 microservice 架構（將模型封裝為 API）

建議流程：建立 branch → 提交修改 → 開 Pull Request，並在 PR 描述中附上本地如何重現的步驟。

---

## 授權

本範例程式碼可自由使用於教學與展示（請保留原作者資訊與變更紀錄）。

---

如果你要我把 README 翻成英文版、或把 README 裡的「如何運行」改為更細緻的步驟（例如 Docker 或 CI/CD），我可以接著新增。祝你使用愉快！
