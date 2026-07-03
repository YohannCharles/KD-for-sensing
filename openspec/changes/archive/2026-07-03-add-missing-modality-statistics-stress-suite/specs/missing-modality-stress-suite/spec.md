## ADDED Requirements

### Requirement: Missing-modality stress suite manifest
系统 MUST 支持缺失模态 stress suite manifest，用于声明 model groups、conditions、severity sweep、difficulty profiles、strict comparability fields 和输出边界。Manifest MUST 支持 smoke、quick 和 formal 三类运行状态。

#### Scenario: 解析 formal stress manifest
- **WHEN** manifest 声明 `suite: missing_modality_stress`
- **THEN** 系统 MUST 标准化 model groups、condition groups、severity values、metric profile、split、seed、label space、difficulty digest 和 output directory
- **AND** 未声明的必要 comparability 字段 MUST 产生 validation error 或 `not_comparable` warning

#### Scenario: smoke manifest 不生成真实 claim
- **WHEN** manifest 使用 synthetic metrics、mock weights、allow_missing_artifacts 或 partial model groups
- **THEN** stress summary MUST 标记为 `mock/smoke`
- **AND** 系统 MUST 不输出真实 claim-ready 状态

### Requirement: Canonical missing-modality conditions
缺失模态 stress suite MUST 提供 canonical condition taxonomy。条件 MUST 至少覆盖 clean/full、single missing、multi missing、only one modality、non-GPS-only、random missing severity 和 unavailable modality。

#### Scenario: 固定缺失条件
- **WHEN** manifest 请求 canonical fixed missing conditions
- **THEN** 系统 MUST 至少生成 `full`、`missing_image`、`missing_radar`、`missing_lidar`、`missing_gps`、`only_gps`、`non_gps_only` 和 `avg_missing` 相关聚合输入
- **AND** 每个 condition MUST 记录 available mask、pattern name、sample count 和 metric rows

#### Scenario: random missing severity
- **WHEN** manifest 请求 random missing severity sweep
- **THEN** 系统 MUST 支持多个 `p_missing` severity values
- **AND** 每个 row MUST 记录 p_missing、seed、ensure_at_least_one、pattern sampling metadata 和 difficulty digest

### Requirement: Input degradation conditions
缺失模态 stress suite MUST 能表达输入退化，而不只表达整模态缺失。输入退化 MUST 通过 shared difficulty pipeline 执行，并 MUST 保持 target、sample id 和 split metadata 不变。

#### Scenario: image degradation sweep
- **WHEN** manifest 请求 image noise、blur、occlusion、night 或 weather severity
- **THEN** 系统 MUST 调用 difficulty pipeline 的 image operator
- **AND** 输出 MUST 记录 severity、operator params、image_valid_mask 或 observability metadata、sample count 和 warning

#### Scenario: GPS noise or async sweep
- **WHEN** manifest 请求 GPS jitter、wrong GPS、dropout、delay 或 async severity
- **THEN** 系统 MUST 调用 difficulty pipeline 的 GPS operator
- **AND** 输出 MUST 记录 GPS source index、delay/dropout/noise metadata、no-future-leak status 和 warning

#### Scenario: unavailable sensing modality
- **WHEN** manifest 请求 radar、LiDAR 或 mmWave unavailable condition
- **THEN** 系统 MUST 通过 missing mask、valid mask 或配置声明的 zero-fill/sentinel 表达不可用
- **AND** 原始 batch target 和 split metadata MUST 保持不变

### Requirement: Stress suite outputs
Stress suite MUST 输出机器可读 condition-level metrics、stress summary、warnings 和 manifest。可选图表和 Markdown 表格 MUST 登记在 manifest 中。

#### Scenario: 写出 stress summary
- **WHEN** stress suite 完成
- **THEN** 输出目录 MUST 包含 condition metrics CSV/JSON、stress summary CSV/JSON、warnings JSON 和 resolved manifest
- **AND** 输出 MUST 包含 strict comparability status、primary metric、per-condition delta 和 aggregate robustness score

#### Scenario: 输出产物边界
- **WHEN** stress suite 写出 tables、figures、cache 或 debug payload
- **THEN** 产物 MUST 位于 ignored `outputs/analysis/missing_modality_stress/` 或用户显式指定目录
- **AND** 源码变更 MUST 不提交真实 stress metrics、figures、checkpoint 或 cache
