## REMOVED Requirements

### Requirement: Near-field beam selection objective
**Reason**: `near_field_beam_selection` 只服务于已退役的 Multimodal-NF 近场三维 codebook beam selection。
**Migration**: 使用当前保留数据集的 beam selection objective；未来若重新引入近场 codebook 任务，应重新提出 capability。

#### Scenario: near-field objective 不再可用
- **WHEN** 用户配置 `experiment.objective: near_field_beam_selection`
- **THEN** 系统 MUST 拒绝该 objective
- **AND** 错误信息 MUST 指出该 objective 随 Multimodal-NF 退役

### Requirement: 三维 codebook target schema
**Reason**: 三维 codebook target schema 绑定 Multimodal-NF Top-5 beam target。
**Migration**: 当前保留目标继续使用各自 target schema。

#### Scenario: codebook target schema 删除
- **WHEN** batch 或配置请求 near-field 三维 codebook target schema
- **THEN** 系统 MUST 不再提供该 schema
- **AND** 当前保留训练流程 MUST 不要求 `beam_triplet_topk` 或 `beam_power_topk`

### Requirement: Near-field beam loss 契约
**Reason**: near-field beam loss 只服务于 Multimodal-NF flattened codebook class。
**Migration**: 使用当前保留 objective 的分类或回归 loss。

#### Scenario: near-field loss 删除
- **WHEN** 用户配置 near-field beam loss
- **THEN** 系统 MUST 拒绝该配置
- **AND** loss registry MUST 不要求 near-field 专属 loss 存在

### Requirement: Near-field beam 指标
**Reason**: near-field Top-K 指标只服务于退役 objective。
**Migration**: 当前保留 metrics 继续由对应 objective metadata 声明。

#### Scenario: near-field 指标不再输出
- **WHEN** 当前保留训练或评估写出 metrics
- **THEN** 系统 MUST 不要求输出 `val_beam_top1`、`val_beam_top3` 或 near-field triplet Top-5 诊断

### Requirement: Near-field objective 运行 metadata
**Reason**: near-field objective runtime metadata 随 Multimodal-NF 删除。
**Migration**: 当前保留 objectives 继续记录各自 runtime metadata。

#### Scenario: near-field metadata 不再写出
- **WHEN** 当前保留训练完成
- **THEN** runtime metadata MUST 不要求记录 codebook shape、flatten order 或 near-field target schema

### Requirement: Near-field beam selection 产物语义
**Reason**: Multimodal-NF near-field codebook run 语义退役。
**Migration**: 当前保留 beam selection run 语义继续由 Raymobtime、DeepSense 或 MMW 对应 specs 约束。

#### Scenario: near-field 产物语义删除
- **WHEN** 用户比较当前保留 run
- **THEN** 系统 MUST 不再生成 Multimodal-NF near-field run metadata
- **AND** 系统 MUST 不把任何当前保留 run 标记为 Multimodal-NF near-field objective
