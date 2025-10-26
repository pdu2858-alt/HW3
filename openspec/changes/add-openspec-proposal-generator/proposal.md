## Why
目前建立 OpenSpec 變更提案需要手動 scaffold 多個檔案與填寫固定結構（proposal.md、tasks.md、spec delta）。這對於頻繁的小型提案會造成重複性工作與格式錯誤風險。

## What Changes
- 新增一個工具/流程支援：自動化變更提案產生器（OpenSpec proposal generator），用以 scaffold `openspec/changes/<change-id>/` 結構並產生初始 `proposal.md`、`tasks.md` 與最小 spec delta 範例。
- 這個變更會新增 `openspec/changes/add-openspec-proposal-generator/` 目錄，包含 proposal、tasks 與一個範例 delta（工具 capability）。

**BREAKING**: 無。

## Impact
- Affected specs: 無會破壞既有規格，但會新增 `tooling` capability 的 ADDED delta。
- Affected code: 主要為專案工具與 CI（如需在 CI 中註冊新的驗證步驟，會更新相關腳本）。

## Rollout
- 初始為文檔與 scaffold 檔案（非執行二進位）。後續可將產生器實作為腳本或 CLI。
