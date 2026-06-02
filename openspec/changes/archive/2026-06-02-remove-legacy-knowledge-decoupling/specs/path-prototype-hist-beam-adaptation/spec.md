## MODIFIED Requirements

### Requirement: Source path loss
source training MUST 支持 beam CE、path semantic CE 和可选 path descriptor regression 的组合 loss。Path loss MUST 只在 batch 中存在合法 path target 时启用。source path loss MUST NOT 要求旧 orthogonality、shared scene confusion 或 private scene preservation loss。

#### Scenario: 有 path_semantic_label 时计算 path CE
- **WHEN** source batch 包含合法 `path_semantic_label` 且 `lambda_path > 0`
- **THEN** training loop MUST 对 `path_logits` 计算 CE loss
- **AND** diagnostics MUST 记录 path loss 和有效样本 coverage

#### Scenario: 有 path_descriptor 时计算 regression
- **WHEN** source batch 包含合法 `path_descriptor`、模型输出 `path_attr_pred` 且 `lambda_path_reg > 0`
- **THEN** training loop MUST 对 `path_attr_pred` 和 `path_descriptor` 计算 SmoothL1 或配置指定 regression loss
- **AND** diagnostics MUST 记录 path descriptor regression MSE 或等价指标

#### Scenario: 保留当前 loss 选项
- **WHEN** 用户运行 V5 coarse、V6 radio-semantic、V8 path 或 hierarchical beam baseline
- **THEN** 系统 MUST 保留当前 radio semantic loss、path loss 和 hierarchical beam loss 配置
- **AND** 系统 MUST NOT 强制启用 path loss
- **AND** 系统 MUST NOT 强制启用旧简单 shared/private 解耦 loss

## ADDED Requirements

### Requirement: Path prototype 不依赖旧解耦 source
Path prototype adaptation MUST NOT 依赖 `v3_decoupled` source-only checkpoint、旧 shared/private source prototype 或旧解耦 loss 才能运行。需要 source 表征或 prototype 时，系统 MUST 从当前合法 source variant 或显式配置的合法 checkpoint 生成。

#### Scenario: 生成 path prototype 时拒绝旧 source
- **WHEN** path prototype generator 收到 source metadata 指向 `v2_shared_private`、`shared_private`、`v3_decoupled` 或 `decoupled`
- **THEN** 系统 MUST 拒绝复用该 source artifact 或将其标记为 retired
- **AND** 错误信息 MUST 提供重新生成当前合法 source prototype 的提示
