## ADDED Requirements

### Requirement: Attention token-read 解释边界
GPS-query 有效性证据包 MUST 将 GPS-query attention 图标记为 query-to-patch token read map，而不得把 raw attention overlay 单独描述为 causal explanation、attribution 或 GPS-query 有效性的主证据。证据包 MUST 在 manifest、report 和 attention 图表 metadata 中记录 attention 来源、shape、token grid、query/time/head 聚合方式、归一化方式和底图来源。

#### Scenario: 写出 token-read 语义 metadata
- **WHEN** evidence package 导出 GPS-query attention patch-grid 或 image overlay
- **THEN** `evidence_manifest.json` MUST 为该图记录 `map_semantics=token_read_map`
- **AND** manifest MUST 记录 `causal_claim=false`、attention source、attention shape、token grid、aggregation method、normalization 和 overlay image source
- **AND** `report.md` MUST 明确说明该图是解释性诊断而非因果归因

#### Scenario: attention 图不得升级 claim
- **WHEN** paired ablation 或 strict comparability 不支持 GPS-query 有效性 claim
- **THEN** 系统 MUST 不因 attention overlay 存在或视觉上聚焦而将 claim 标记为 `supported`
- **AND** 系统 MUST 将 attention 相关结论放入 `interpretive` 或 `caveat`

### Requirement: Attention faithfulness 诊断
GPS-query 有效性证据包 MUST 支持 opt-in attention faithfulness 诊断。系统 MUST 基于 attention token-read score 选择 top-attention patch、low-attention patch 和 deterministic random patch，对输入 image 或 token 进行相同预算的遮挡或替换，并比较目标 logit、target margin、Top-k、DBA contribution 或等价指标变化。

#### Scenario: 导出 faithfulness 表
- **WHEN** evidence config 启用 attention faithfulness 且模型、attention map 和可遮挡输入可用
- **THEN** 系统 MUST 写出 `tables/attention_faithfulness.csv`
- **AND** 每行 MUST 包含 model、sample id、patch selection group、patch count 或 patch ratio、occlusion strategy、seed、baseline metric、occluded metric、absolute delta 和 faithfulness status

#### Scenario: 比较 top、low 和 random patch
- **WHEN** 系统对同一样本执行 attention faithfulness 诊断
- **THEN** 系统 MUST 至少比较 `top_attention`、`low_attention` 和 `random` 三类 patch selection
- **AND** 每类 selection MUST 使用相同 patch 数或相同 patch ratio
- **AND** random selection MUST 使用记录在 manifest 中的 deterministic seed

#### Scenario: faithfulness 输入不可用降级
- **WHEN** attention map 可用但没有可遮挡的 image tensor、raw image path 或 token-level fallback
- **THEN** 系统 MUST 跳过该样本的 faithfulness 诊断
- **AND** 系统 MUST 继续导出 attention summary、paired delta 和 report
- **AND** manifest 和 report MUST 记录 skipped reason

### Requirement: Faithfulness-aware claim gate
GPS-query 有效性 claim gate MUST 将 attention faithfulness 结果作为解释性证据门控项。Claim gate MUST 继续以 strict comparability、paired delta、clean regression、P0-P5 robustness 和 case coverage 为主证据；attention faithfulness 只能支持解释性结论，不能单独支持有效性 claim。

#### Scenario: faithfulness 支持解释性 claim
- **WHEN** strict paired delta 支持 GPS-query 有效性且 top-attention 遮挡造成的指标下降稳定大于 low-attention 和 random 遮挡
- **THEN** claim gate MAY 将 attention 解释项标记为 `supported`
- **AND** report MUST 将该结论写入 `interpretive`，并引用 `attention_faithfulness.csv`

#### Scenario: faithfulness 不通过时降级
- **WHEN** top-attention 遮挡不比 low-attention 或 random 遮挡造成更大指标下降
- **THEN** claim gate MUST 将 attention 解释项标记为 `insufficient` 或 `exploratory`
- **AND** report MUST 明确说明 token-read map 未通过 faithfulness 检查

#### Scenario: paired evidence 不足时阻止 supported claim
- **WHEN** faithfulness 诊断通过但 paired ablation 不可比、样本不足或 clean/P0 delta 不支持
- **THEN** claim gate MUST 不将 GPS-query 有效性 claim 标记为 `supported`
- **AND** attention faithfulness 结果 MUST 仅作为 caveat 或 exploratory diagnostic 输出
