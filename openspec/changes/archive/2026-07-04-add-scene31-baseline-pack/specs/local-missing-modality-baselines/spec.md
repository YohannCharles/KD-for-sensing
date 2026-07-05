## ADDED Requirements

### Requirement: AMR-lite local baseline
系统 MUST 提供 AMR-lite local missing-modality baseline，用于表达 available modality masking、missing modality imputation 和轻量 channel/modality attention。该 baseline MUST 标记为 local experimental baseline，不得声明为完整外部论文复现。

#### Scenario: AMR-lite 构建
- **WHEN** 配置声明 `amr_lite`
- **THEN** 系统 MUST 能构建支持 image、LiDAR、radar 和 GPS modality feature 的轻量 fusion 模型或组件
- **AND** 缺失 modality MUST 使用 `zero`、`mean_feature` 或 `learnable_token` imputation 之一，默认 MUST 支持 `learnable_token`
- **AND** 模型 MUST 使用 availability/missing mask 参与 gate 或 attention 计算

#### Scenario: AMR-lite gate stats
- **WHEN** AMR-lite 完成一个训练 epoch 或可用诊断阶段
- **THEN** run 目录 MUST 写入 `amr_lite_gate_stats.csv`
- **AND** CSV MUST 包含 `epoch`、`pattern`、`modality`、`mean_gate` 和 `std_gate`

#### Scenario: AMR-lite 防泄漏
- **WHEN** 某 modality 在训练或评估 batch 中被标记为缺失
- **THEN** AMR-lite MUST NOT 使用该 modality 的真实 feature 作为可见输入
- **AND** imputation 和 gate 计算 MUST 只依赖缺失 token、可见 feature 和 mask metadata

### Requirement: AMBER-lite baseline-pack training comparison
AMBER-lite 在 baseline pack 中 MUST 支持 natural 或 randomdrop training 与 pattern-balanced exposure training 的公平对照。该对照 MUST 继续保持 AMBER-lite local-lite scope，不升级为 AMBER full 或官方复现。

#### Scenario: AMBER-lite pattern-balanced 对照
- **WHEN** baseline pack 生成 AMBER-lite run matrix
- **THEN** 矩阵 MUST 包含 `amber_lite_uniform_es40_seed1/2/3`
- **AND** 矩阵 MUST 包含 `amber_lite_natural_es40_seed1/2/3` 或 `amber_lite_randomdrop_subset_es40_seed1/2/3`

#### Scenario: AMBER-lite 参数统计
- **WHEN** AMBER-lite baseline 被 summary 读取或模型可构建
- **THEN** 系统 MUST 统计或保留 `total_params`、`trainable_params` 和 `extra_params_vs_proto`
- **AND** 参数来源 MUST 区分真实 module 统计和 unavailable

### Requirement: FeatureMod-lite local baseline
系统 MUST 支持将 FeatureMod-lite 作为可选运行组纳入 baseline pack，用于对比 missing-modality feature adaptation。FeatureMod-lite MUST 使用 missing modalities 作为 condition，不得复用失败的 PatternFiLM d8 作为同一方法换名。

#### Scenario: FeatureMod-lite 构建
- **WHEN** 配置声明 `featuremod_lite`
- **THEN** 系统 MUST 使用 missing modalities condition 生成轻量 adapter 或 affine feature correction
- **AND** 默认 adapter 维度 MUST 保持小参数量，例如 `adapter_dim=16`

#### Scenario: FeatureMod-lite baseline pack run
- **WHEN** 用户选择 baseline pack `featuremod` group
- **THEN** runner MUST 尝试运行 `featuremod_lite_uniform_es40_seed1/2/3`
- **AND** 若 FeatureMod-lite 未实现或只完成 quick screen，summary MUST 标记 skipped 或 quick_screen，不得纳入最终多 seed claim

### Requirement: Baseline pack 参数量比较
本地缺失模态 baseline 被纳入 baseline pack summary 时 MUST 提供可审计参数量字段。参数比较 MUST 支持 proto、AMR-lite、AMBER-lite 和 FeatureMod-lite。

#### Scenario: 参数字段
- **WHEN** baseline pack summary 写出 `params_comparison.csv`
- **THEN** 每个可用 method MUST 包含 `total_params`、`trainable_params` 和 `extra_params_vs_proto`
- **AND** 无法确认参数量的历史或缺失 run MUST 使用空值并记录 warning，不得伪造数值
