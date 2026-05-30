## ADDED Requirements

### Requirement: Metric horizon aggregation consistency
训练验证、force-mask subset 验证和 standalone evaluate MUST 对 beam Top-K、ADBA/DBA 和公开 top-level scalar 使用同一套 selected metric horizons。配置或 runtime 解析出的 `metric_horizons` MUST 被记录在 metrics metadata 中，subset top-level scalar MUST NOT 回退到 first valid slot 口径。

#### Scenario: subset top1 使用 selected horizons
- **WHEN** 配置选择 `metric_horizons=[2,4,6]` 或等价 horizon 集合
- **THEN** 普通 validation 的 top-level Top-1 MUST 基于这些 selected horizons 聚合
- **AND** force-mask subset validation 的 top-level `top1` 或等价 scalar MUST 使用同一 selected horizon 聚合
- **AND** subset validation MUST NOT 使用 first valid slot 作为 top-level `top1`

#### Scenario: standalone evaluate 记录同一口径
- **WHEN** 用户通过 standalone evaluate 运行同一配置
- **THEN** evaluate metrics/report MUST 记录实际使用的 `metric_horizons`
- **AND** Top-K 与 DBA/ADBA top-level scalar MUST 与训练验证使用同一 horizon 选择规则
- **AND** 若输出逐 horizon 诊断，诊断字段 MUST 与 top-level 聚合字段可区分

#### Scenario: 未配置 horizons 使用统一默认
- **WHEN** 配置没有显式设置 `metric_horizons`
- **THEN** validation、subset validation 和 evaluate MUST 使用同一个默认 horizon 集合
- **AND** metrics metadata MUST 记录默认来源或等价说明
