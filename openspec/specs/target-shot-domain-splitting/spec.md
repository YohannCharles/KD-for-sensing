# target-shot-domain-splitting Specification

## Purpose
TBD - created by archiving change add-target-shot-geometry-residual-foundations. Update Purpose after archive.
## Requirements
### Requirement: 可配置 source-target domain 定义
系统 MUST 支持通过配置定义多场景/多天气 beam prediction 的 source domain 和 target domain。domain 类型 MUST 至少支持 `scenario`、`weather`、`scenario_weather` 和 `town_scenario_weather`；系统 MUST 使用 sample metadata 中的明确字段构造 domain key，不得依赖不可解析的 sample_id 字符串猜测 domain。

#### Scenario: 按 scenario_weather 选择 domain
- **WHEN** 配置设置 `split.domain_type: scenario_weather`、`split.source_domains` 和 `split.target_domains`
- **THEN** split builder MUST 从样本 metadata 的 scenario 与 weather/condition 字段构造 domain key
- **AND** 只把匹配 source domain 的样本放入 source split
- **AND** 只把匹配 target domain 的样本放入 target split

#### Scenario: domain 字段缺失
- **WHEN** 配置要求 `town_scenario_weather` 但样本 metadata 缺少 town、scenario 或 weather/condition 中任一字段
- **THEN** split builder MUST 拒绝生成 split
- **AND** 错误信息 MUST 包含缺失字段名、dataset type 和可执行修复提示

### Requirement: 5% target-shot target 拆分
系统 MUST 将 target domain 确定性拆分为 `target_labeled`、可选 `target_unlabeled` 和 `target_test`。`target_labeled` MUST 只从 target adaptation pool 中采样，默认目标比例为 `target_label_fraction=0.05`；`target_test` MUST 与 source、target_labeled 和 target_unlabeled 无 sample id 交集。

#### Scenario: target_labeled 比例可复现
- **WHEN** 用户使用相同输入 manifest/CSV、domain 配置、`target_label_fraction: 0.05`、selection strategy 和 seed 生成 split 两次
- **THEN** 两次生成的 `target_labeled` sample ids MUST 完全一致
- **AND** `target_labeled` 数量 MUST 等于可用 target adaptation pool 的 5% 四舍五入策略结果或 metadata 中声明的最小样本修正规则

#### Scenario: target split 无交集
- **WHEN** split builder 完成 source、target_labeled、target_unlabeled 和 target_test 拆分
- **THEN** 任意两个 split 的 sample id 集合 MUST 无交集
- **AND** 对序列窗口数据，split metadata MUST 继续记录 frame/window overlap 与 guard band leakage diagnostics

### Requirement: target labeled subset 分层采样
系统 MUST 支持 `random`、`stratified_by_beam`、`stratified_by_geo_sector` 和 `stratified_by_weather` target labeled selection strategy。无法满足某个分层桶的最小样本数时，系统 MUST 记录降级原因并保持整体采样可复现。

#### Scenario: 按 beam 分层采样
- **WHEN** 配置设置 `split.target_label_selection: stratified_by_beam`
- **THEN** split builder MUST 基于 target adaptation pool 中的 beam label 分布选择 target_labeled
- **AND** sampling manifest MUST 记录每个 beam 的候选数、选中数和 seed

#### Scenario: 按 geo_sector 分层但 geometry 不可用
- **WHEN** 配置设置 `split.target_label_selection: stratified_by_geo_sector` 且 target adaptation pool 缺少 geo_sector
- **THEN** split builder MUST 拒绝该策略或按配置声明的 fallback 策略降级
- **AND** split metadata MUST 记录 fallback reason

### Requirement: split artifact 持久化与复用
系统 MUST 将 split indices/sample ids、配置摘要、输入 fingerprint、domain metadata、label histogram、weather/scenario histogram、target labeled sampling manifest 和 leakage diagnostics 写入 JSON 或 NPZ artifact。后续训练、适配、评估和诊断 MUST 能通过 artifact 复用同一 split。

#### Scenario: split artifact 匹配当前输入
- **WHEN** 用户传入已有 split artifact 且输入 sample ids、配置摘要和 fingerprint 匹配
- **THEN** 系统 MUST 复用 artifact 中的 split
- **AND** 不得重新随机采样 target_labeled

#### Scenario: split artifact 不匹配
- **WHEN** 已有 split artifact 的 sample id fingerprint、target domain、seed 或 target_label_fraction 与当前配置不匹配
- **THEN** 系统 MUST 拒绝复用该 artifact
- **AND** 错误信息 MUST 指出不匹配字段，并提示 regenerate 或 overwrite

### Requirement: target_unlabeled 监督字段防泄漏
系统 MUST 将 target_unlabeled 标记为无监督 target subset。训练 loss、adaptation、threshold selection、temperature fitting、prototype update 和 early stopping MUST NOT 访问 target_unlabeled 的 beam、residual、beam_power、CSI/channel、path 或 radio supervision 字段。

#### Scenario: target_unlabeled 访问 beam supervision 失败
- **WHEN** target_unlabeled batch 被用于 adaptation 且 loss 代码尝试读取 beam 或 residual label 作为监督
- **THEN** runtime guard MUST raise error
- **AND** 错误信息 MUST 包含 split、subset、field name、label fraction 和修复提示

#### Scenario: target_labeled 允许 beam supervision
- **WHEN** batch 来自 `target_labeled` 且 `target_label_fraction > 0`
- **THEN** supervised beam 或 residual loss MAY 读取该 batch 的 beam/residual label
- **AND** metadata MUST 记录该监督只来自 target_labeled subset

