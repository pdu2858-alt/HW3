## ADDED Requirements
### Requirement: Proposal generator scaffold
專案 SHALL 提供一個可以產生 OpenSpec 變更提案 scaffold 的機制或範例。此 scaffold 應包含：
- `proposal.md`
- `tasks.md`
- `specs/<capability>/spec.md`（至少一個範例 delta）

#### Scenario: Generate minimal scaffold
- **WHEN** a maintainer needs to create a new change proposal
- **THEN** they SHALL be able to create a directory `openspec/changes/<change-id>/` containing `proposal.md`、`tasks.md`、以及一個範例 spec delta，便於後續編輯與驗證。
