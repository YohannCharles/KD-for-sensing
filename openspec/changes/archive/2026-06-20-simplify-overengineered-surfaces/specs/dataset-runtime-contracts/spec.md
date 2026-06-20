## ADDED Requirements

### Requirement: Dataset runtime 契约不要求未接入通用 framework
Dataset runtime contract MAY 由当前 dataset 实现、轻量 row 类型、metadata helper 和 data factory 组合满足。系统 MUST 不要求保留没有当前调用方的通用 `RuntimeDataset`、`SampleIndex`、`ModalityAdapter`、`TargetProvider` 或 index writer 框架模块；若这些符号无 current surface 消费，项目 MAY 删除它们。

#### Scenario: flat sample 行为由现有 dataset 满足
- **WHEN** 当前保留 dataset 构建 dataloader 并返回样本
- **THEN** flat sample keys、target fields、sample id、split metadata 和 enabled modality behavior MUST 保持与现有 dataset contract 兼容
- **AND** 系统 MUST 不要求通过独立 `RuntimeDataset` wrapper 才能满足该行为

#### Scenario: 保留仍被消费的轻量 row 类型
- **WHEN** `SampleRow` 或等价轻量 row 类型仍被 target-shot split、metadata 或测试消费
- **THEN** 本 change MUST 保留该类型或迁移到实际 owner 模块
- **AND** 删除未接入 framework MUST 不破坏 target-shot split artifact 读取

#### Scenario: 删除未消费 adapter framework
- **WHEN** 通用 sample index、modality adapter、target provider 或 index writer 没有内部调用、公开导出、配置、docs 或 current OpenSpec 消费
- **THEN** 本 change MAY 删除这些符号和只服务它们的测试
- **AND** dataset runtime metadata、sensitive field guard 和 difficulty metadata 契约 MUST 继续由现有 runtime 路径满足
