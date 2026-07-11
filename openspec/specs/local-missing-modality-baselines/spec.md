# local-missing-modality-baselines Specification

## Purpose
约束 AMBER-lite、RMBP-MM 和 TII-VLRG-style 缺失模态 baseline 在本仓库实验场景中的本地训练、验证、配置、metadata 和 claim 边界。
## Requirements
### Requirement: AMR-lite local baseline
系统 MUST 为 protected Scene31-34 external-lite workflow 保留 AMR-lite local missing-modality component，用于表达 available modality masking、missing modality imputation 和轻量 channel/modality attention。该 baseline MUST 标记为 local experimental baseline，不得声明为完整外部论文复现，也不得扩大为 standalone baseline-pack product。

#### Scenario: AMR-lite 构建
- **WHEN** Scene31-34 generated config 声明 `amr_lite`
- **THEN** 系统 MUST 能构建支持 image、LiDAR、radar 和 GPS modality feature 的轻量 fusion component
- **AND** 缺失 modality MUST 使用 current config 声明的 imputation，availability/missing mask MUST 参与 gate 或 attention

#### Scenario: AMR-lite 防泄漏
- **WHEN** 某 modality 在训练或评估 batch 中被标记为缺失
- **THEN** AMR-lite MUST NOT 使用该 modality 的真实 feature 作为可见输入
- **AND** imputation 和 gate 计算 MUST 只依赖缺失 token、可见 feature 和 mask metadata

#### Scenario: AMR-lite diagnostics 服从 Scene31-34 owner
- **WHEN** AMR-lite 进入 Scene31-34 fresh evaluation
- **THEN** output MUST 使用 Scene31-34 current maskfix/status contract
- **AND** capability MUST 不要求独立 epoch gate CSV 或 standalone diagnostics runner

### Requirement: Local missing-modality capability 只保留 AMR-lite supporting contract
该 capability MUST 分类为 supporting，并只保护 Scene31-34 runner 与 generated configs 仍消费的 AMR-lite component、missing-mask/gate 和 local-baseline claim boundary。AMBER-lite、AMBER full 与 AMR-Net MUST 分别由其独立 current specs 管理；legacy baseline-pack、FeatureMod 和旧 maskfix wrappers MUST 不借此恢复。

#### Scenario: Scene31-34 构建 AMR-lite
- **WHEN** protected Scene31-34 generator 生成 `amr_lite` natural/uniform config
- **THEN** current component registry MUST 能构建该 local AMR-lite baseline
- **AND** missing modality MUST 不泄漏真实 feature，availability mask MUST 参与 gate/attention

#### Scenario: Supporting lifecycle 不恢复 baseline pack
- **WHEN** lifecycle、scripts 和 configs 被枚举
- **THEN** capability MUST 标记为 `supporting`
- **AND** FeatureMod、旧 Scene31 baseline pack、RMBP/TII 聚合入口和 retired maskfix wrappers MUST 不存在

