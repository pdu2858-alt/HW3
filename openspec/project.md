# Project Context

## Purpose
本專案 `HW3` 使用 OpenSpec 驅動規格（spec-driven）的方法管理功能變更與需求，目標是透過結構化的變更提案、規格 delta 與驗證工具，讓功能新增與變更更可追蹤、可驗證且易於審查。

此文件提供本專案的技術棧、開發與提交慣例，以及對 AI 助手（如本 repo 的協作機制）有幫助的背景資訊。

## Tech Stack
- 開發環境: Ubuntu dev container（已在 repository 開發環境中設定）
- 主要工具: `openspec` CLI（用於列出/驗證/封存變更與規格）
- 版本控制: Git + GitHub (owner: pdu2858-alt)
- 腳本與自動化: Bash、Makefile（視需求）
- 可選（視實作需求）: Node.js / Python（用於工具或驗證腳本）

## Project Conventions

### Code & Spec Style
- 所有規格使用 Markdown，並遵循 `openspec/AGENTS.md` 的格式要求（尤其是 `#### Scenario:` 的格式與 `## ADDED|MODIFIED|REMOVED Requirements` 分節）。
- 需求描述使用 SHALL/MUST 做為規範語句（避免使用 should/may，除非刻意為非強制）。
- 程式碼風格依語言使用常見 formatter（例如 ESLint+Prettier for JS/TS，black for Python）；若新增語言或工具，請在本檔補充具體設定檔位置。

### Architecture Patterns
- 採取單一職責的 capability 切分：每個 capability 放在 `openspec/specs/<capability>/spec.md`。
- 變更採用三階段流程（proposal → implement → archive），詳見 `openspec/AGENTS.md`。

### Testing Strategy
- 規格層面：每個需求至少一個 `#### Scenario:`，以便自動或人工驗證。
- 實作層面：每個變更的 `tasks.md` 應包含單元測試與整合測試項目；CI 應在 PR 阶段跑測試與 `openspec validate <change-id> --strict`。

### Git Workflow
- 分支策略：每個變更建立 feature 分支，命名與 change-id 保持一致或以 `ch/<change-id>` 為前綴。
- commit 訊息：使用簡單的 verb-led 開頭（例如 `feat: add X`, `fix: ...`, `docs: ...`），並在 PR 描述中提到相關 `openspec` 變更目錄（例：`openspec/changes/add-foo/`）。
- PR 流程：在 PR 前確保 `openspec validate <change-id> --strict` 通過；請在 PR 描述中附上 `proposal.md` 摘要與 `tasks.md` checklist。

## Domain Context
- 本 repo 目前採用 OpenSpec 來管理功能規格，規格即權威來源（single source of truth）。
- 主要關注點在於變更管理流程，而非單一應用程式語言或框架。

## Important Constraints
- 任何影響現有規格（MODIFIED/REMOVED）的變更必須有完整的規格 delta 並經過嚴格驗證與審查。
- 變更提案不得在未審查前直接實作（Stage 1 → Stage 2 的規則必須遵守）。

## External Dependencies
- `openspec` CLI 與其驗證工具（請確保 CI runner 有安裝並可執行）
- GitHub（PR、CI、審查流）

## Contact / Ownership
- Repo owner: `pdu2858-alt`
- 若有疑問，優先在相關 change 的 `proposal.md` 下留言 or 開 issue 討論。

## Assumptions
- 如果專案需要更明確的技術棧（例如前端框架或後端語言），請提供或允許我提出合理的選擇（我會在 `project.md` 補充具體建議）。

---

上面內容為初版填寫；如果你要我改成英文、補充具體 CI 設定或加入更多實作細節（例如 Node.js/pytest 範例），請告訴我，我會繼續更新。
