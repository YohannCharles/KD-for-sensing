## ADDED Requirements

### Requirement: 退役 Hist 输出候选分类
清理系统 MUST 能将用户明确退役的 Hist/HiST-Beam、P3、V8/V9 probe、image-only Hist、history-anchor Hist、debug、smoke 和 plan-check 输出识别为本地运行产物候选。每个候选 MUST 记录稳定规则 ID、匹配原因、风险等级、大小、mtime 和保护状态。

#### Scenario: Hist 输出进入候选
- **WHEN** 扫描发现 `outputs/hist_beam_loso`、`outputs/history_anchor_*`、`outputs/image_only_legal_*`、`outputs/p3_v8_*` 或 `outputs/v9_*`
- **THEN** manifest MUST 将其列为退役 Hist 输出候选或需要人工确认候选
- **AND** manifest MUST 记录这些目录与已退役 Hist 研究线的关系

#### Scenario: debug 和 plan-check 输出进入低风险候选
- **WHEN** 扫描发现 `outputs/_debug_*`、`outputs/*_plan_check*` 或短生命周期 smoke 输出
- **THEN** manifest MUST 将其列为低风险或中风险清理候选
- **AND** manifest MUST 记录候选是否包含 checkpoint、metrics 或 source config

### Requirement: Hist 字符串不得作为唯一删除条件
清理系统 MUST 不得仅因路径包含 `hist` 字符串就删除产物。候选规则 MUST 结合 workflow 名称、run metadata、目录语义、退役清单或用户明确规则，避免误删历史窗口 baseline 或当前主线诊断。

#### Scenario: GPS history-window baseline 需要复核
- **WHEN** 扫描发现 `gps_window_*hist2` 或其它仅表示历史窗口长度的目录
- **THEN** manifest MUST 不得仅凭 `hist2` 将其归为 HiST-Beam 删除候选
- **AND** 如需删除，候选原因 MUST 来自 stale、debug、duplicate、用户显式模式或其它非裸字符串规则
