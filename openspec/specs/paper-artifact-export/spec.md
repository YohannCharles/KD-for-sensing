# paper-artifact-export Specification

## Purpose
定义从已审阅 claim、ledger 或 summary 导出论文表格草稿、figure-data 和 export manifest 的边界，保证 provenance、claim status 和 ignored output root 可追踪。
## Requirements
### Requirement: Paper artifact export
系统 MUST 提供 paper artifact export 能力，用于从已审阅 claim、ledger 或 summary 中生成论文表格/图草稿。Export MUST 保留每行 claim status 和 provenance，并 MUST 不提交真实生成图表或最终论文产物到源码。

#### Scenario: 生成主表草稿
- **WHEN** 用户运行 paper export 并指定 reviewed claim registry 或 ledger 输入
- **THEN** 系统 MUST 生成 Markdown、CSV 或 LaTeX 中至少一种表格草稿
- **AND** 每行 MUST 包含 method、dataset/split、metric、value、claim_status、provenance 和 caveat

#### Scenario: pending rows 默认不进主表
- **WHEN** 输入包含 pending、mock/smoke、historical ablation、upper-bound 或 not_comparable rows
- **THEN** 系统 MUST 默认将这些 rows 排除出 main table
- **AND** 若用户显式包含它们，输出 MUST 保留状态列和 caveat

#### Scenario: 写出 export manifest
- **WHEN** paper export 完成
- **THEN** 输出目录 MUST 包含 `paper_export_manifest.json` 或等价 manifest
- **AND** manifest MUST 记录输入文件、输入 claim ids、过滤规则、输出文件、warnings、generated_at 和 git commit 或 unavailable 状态

### Requirement: Paper figure data export
系统 MUST 支持从 stress summary、pattern summary 或 claim rows 导出论文图所需的数据文件。图数据 MUST 与最终 PNG/SVG/PDF 分离。

#### Scenario: 导出 stress curve 数据
- **WHEN** 输入包含 stress suite condition-level metrics
- **THEN** 系统 MUST 能导出 stress curve CSV/JSON
- **AND** 输出 MUST 包含 condition、severity、method、metric、mean/std 或 CI、claim status 和 caveat

#### Scenario: 导出 pattern heatmap 数据
- **WHEN** 输入包含 missing pattern matrix
- **THEN** 系统 MUST 能导出 pattern heatmap CSV/JSON
- **AND** 输出 MUST 保留 pattern definition、available mask、metric profile 和 sample count

### Requirement: Paper artifact output boundary
Paper export 生成的 tables、figure data、plots 和 manifest MUST 默认写入 ignored `outputs/paper_artifacts/` 或用户显式指定路径。

#### Scenario: 输出到 ignored 目录
- **WHEN** 用户未显式指定输出目录
- **THEN** paper export MUST 写入 `outputs/paper_artifacts/`
- **AND** 源码变更 MUST 不包含生成的真实表格、图、PDF、PNG、SVG 或 notebook output

### Requirement: 主表导出 gate 硬排除不合格 claim
Paper artifact export MUST 默认排除 `pending`、`mock/smoke`、`historical ablation`、`upper-bound`、`blocked official reproduction`、`not_comparable`、`unverified` 和 `candidate_only=true` 的行进入正式主表。被排除行如被导出，MUST 进入 excluded report 或显式标注的 appendix draft，并 MUST 保留 status 和 caveat。

#### Scenario: pending rows 不进入主表
- **WHEN** 输入 claim registry 或 ledger 包含 pending、mock/smoke、upper-bound 或 not_comparable 行
- **THEN** paper export 主表 MUST 不包含这些行
- **AND** excluded report MUST 说明排除原因和 claim id

#### Scenario: 人工覆盖需要显式参数
- **WHEN** 用户显式要求导出非正式 appendix 或 diagnostics table
- **THEN** 系统 MUST 仅将不合格状态行导出到显式标注的非正式 appendix 或 diagnostics table
- **AND** 输出表 MUST 显示 claim status 和 caveat，不得伪装成正式主表

### Requirement: Paper main table 使用 reviewed allowlist 与必填 schema
paper exporter MUST 只允许明确 reviewed 的状态进入 main table，并 MUST 在纳入前验证 claim schema 必填字段。未知、空、pending、not_comparable、invalidated、mock/smoke、historical、upper-bound、blocked、unverified 或 candidate-only 行 MUST 默认进入 excluded report。

#### Scenario: 空或未知状态
- **WHEN** claim status 为空或不在 reviewed allowlist
- **THEN** row MUST NOT 进入 main table
- **AND** excluded report MUST 记录 `status_not_reviewed`

#### Scenario: Reviewed 状态但字段缺失
- **WHEN** status 在 allowlist 中但 method、dataset/split、metric/value、统计或 provenance 必填字段缺失
- **THEN** row MUST NOT 进入 main table
- **AND** excluded report MUST 列出 missing fields

#### Scenario: 完整 reviewed claim
- **WHEN** claim status 在 allowlist、candidate flag 为 false 且必填字段完整
- **THEN** exporter MUST 允许 row 进入 main table
- **AND** output MUST 保留 status、provenance 和 caveat

