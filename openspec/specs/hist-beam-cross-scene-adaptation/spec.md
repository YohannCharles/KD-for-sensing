# hist-beam-cross-scene-adaptation Specification

## Purpose
TBD - created by archiving change add-hist-beam-cross-scene-adaptation. Update Purpose after archive.
## Requirements
### Requirement: HiST-Beam 模型变体配置
系统 MUST 提供可通过配置和模型注册表构建的 HiST-Beam fusion 模型能力，用于 DeepSense6G 跨场景快速验证。配置 MUST 能选择 flat source-only、hierarchical source-only、shared-private、decoupled shared-private、adapter-only、adapter+prototype 和 full fine-tuning baseline 变体，并 MUST 默认使用 `image`、`radar`、`gps` 三模态。

#### Scenario: 构建 flat source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v0_flat` 或等价 flat 模式
- **THEN** 系统 MUST 构建普通 64 类 beam classifier
- **AND** 模型输出 MUST 继续兼容现有 beam Top-K 评估流程

#### Scenario: 构建 hierarchical source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v1_hierarchical`
- **THEN** 系统 MUST 构建 coarse head 和 fine head
- **AND** 系统 MUST 不启用 shared/private 解耦 loss 或 target adapter

#### Scenario: 构建 shared-private 解耦变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v3_decoupled`
- **THEN** 模型 MUST 产生 shared representation、private representation、coarse logits、fine logits 和 beam-level prediction
- **AND** 训练 MUST 能启用 orthogonality、shared scene confusion 和 private scene preservation loss

#### Scenario: 构建 adapter 和 full fine-tuning 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v4_adapter`、`v5_adapter_proto` 或 `v6_full_finetune`
- **THEN** 系统 MUST 从 source checkpoint 初始化 target adaptation run
- **AND** 系统 MUST 按变体选择 adapter 训练、prototype alignment 或全量 fine-tuning 策略

### Requirement: 层次化 beam label 与输出契约
HiST-Beam MUST 支持将 64 类 beam label 拆分为 coarse group 和 fine offset。`group_size` MUST 可配置，快速验证默认值 MUST 为 8；当 `num_classes=64` 且 `group_size=8` 时，coarse group 数 MUST 为 8。

#### Scenario: 生成 coarse 和 fine label
- **WHEN** 输入 beam label 为合法的 64 类整数，且 `group_size=8`
- **THEN** 系统 MUST 将 coarse label 计算为 `beam // 8`
- **AND** 系统 MUST 将 fine offset 计算为 `beam % 8`

#### Scenario: 拒绝不可整除 group size
- **WHEN** 用户配置的 `num_classes` 不能被 `group_size` 整除
- **THEN** 系统 MUST 拒绝构建 HiST-Beam 配置
- **AND** 错误信息 MUST 包含 `num_classes`、`group_size` 和可执行的修复提示

#### Scenario: 输出 beam-level prediction
- **WHEN** hierarchical head 完成 forward
- **THEN** 模型 MUST 输出 coarse logits、fine logits 和 beam-level logits 或 log probabilities
- **AND** beam-level 输出 MUST 能用于 Top-1、Top-3 和 Top-5 指标

#### Scenario: 保持 horizon 维兼容
- **WHEN** 配置 `model.num_pred` 大于 1
- **THEN** HiST-Beam 输出 MUST 保持 `[B, H, C]` 的 beam-level 形状
- **AND** coarse/fine diagnostics MUST 与同一 horizon 对齐

### Requirement: Shared-private 表示解耦
HiST-Beam shared branch MUST 主要服务 coarse semantics，private branch MUST 主要服务 scene-specific refinement。启用 shared-private 时，coarse head MUST 只读取 shared representation，fine head MUST 读取 shared 与 private 的组合。

#### Scenario: coarse head 不读取 private representation
- **WHEN** 模型启用 shared-private hierarchical 模式
- **THEN** coarse head 的输入 MUST 只来自 shared representation
- **AND** private representation 的变化 MUST 不直接作为 coarse head 输入

#### Scenario: private branch 保留 scene 信息
- **WHEN** 配置启用 private scene preservation loss
- **THEN** 模型 MUST 输出 private scene classifier logits
- **AND** private branch 参数 MUST 接收该 loss 的梯度，除非配置显式关闭该 loss

#### Scenario: shared branch 使用 scene confusion
- **WHEN** 配置启用 shared scene confusion loss
- **THEN** 模型 MUST 输出 shared scene classifier logits
- **AND** shared branch MUST 通过 GRL 或等价反向机制降低 scene 可辨识性

### Requirement: HiST-Beam 训练 loss
系统 MUST 在显式启用 HiST-Beam 时计算层次化 loss、flat auxiliary loss、orthogonality loss、shared scene confusion loss 和 private scene preservation loss。普通非 HiST 配置 MUST 不受这些 loss 影响。

#### Scenario: 计算 hierarchical loss
- **WHEN** 训练 hierarchical 变体
- **THEN** 系统 MUST 对 coarse logits 计算 coarse CE
- **AND** 系统 MUST 只在真实 coarse group 对应的 fine logits 上计算 fine CE
- **AND** 系统 MUST 按配置权重合成 hierarchical loss

#### Scenario: 计算 flat auxiliary loss
- **WHEN** 配置 `lambda_flat` 大于 0
- **THEN** 系统 MUST 从 beam-level 输出计算 beam class 辅助 loss
- **AND** 该 loss MUST 参与总 loss 以约束最终 beam prediction

#### Scenario: HiST loss 不影响普通模型
- **WHEN** 用户运行非 HiST-Beam 模型或未启用 HiST loss 的配置
- **THEN** 训练流程 MUST 使用既有 beam loss 语义
- **AND** 系统 MUST 不要求模型输出 coarse/fine/shared/private diagnostics

### Requirement: Source prototype artifact
完成 source training 后，系统 MUST 能基于 source train split 生成 coarse prototype artifact。artifact MUST 至少包含 shared prototype、private prototype、每个 coarse group 的样本计数和生成配置 metadata。

#### Scenario: 保存 source prototypes
- **WHEN** HiST-Beam source training 完成且配置启用 prototype 保存
- **THEN** 系统 MUST 在运行目录保存 prototype artifact
- **AND** artifact MUST 记录 `group_size`、`num_groups`、source scenes、target scene、seed 和样本计数

#### Scenario: 空 group prototype 可诊断
- **WHEN** 某个 coarse group 在 source train split 中没有样本
- **THEN** prototype artifact MUST 记录该 group 的 count 为 0
- **AND** target adaptation MUST 不把该 group 用作高置信 prototype 目标

### Requirement: Target adapter adaptation
系统 MUST 支持从 source checkpoint 启动 target adaptation。Adapter-only 和 Adapter+Prototype 变体 MUST 冻结 source backbone 和 shared/coarse 主干，只训练配置允许的少量参数；full fine-tuning baseline MUST 能在相同 target labeled samples 上更新全部参数。

#### Scenario: Adapter 训练参数受限
- **WHEN** 用户运行 adapter-only 或 adapter+prototype adaptation
- **THEN** 系统 MUST 冻结 image/radar/gps encoder、fusion module、shared branch、coarse head 和原始 private branch
- **AND** 系统 MUST 只训练 private adapter、允许的 fine head 参数、可选 LayerNorm affine 参数和配置允许的 prototype 参数

#### Scenario: Adapter 初始等价 source model
- **WHEN** private adapter 初始化完成且尚未训练
- **THEN** adapter 输出 MUST 与未启用 adapter 的 private representation 等价或数值上等价
- **AND** source-only evaluation MUST 不因创建 adapter 模块而改变预测

#### Scenario: Full fine-tuning 更新全部参数
- **WHEN** 用户运行 full fine-tuning baseline
- **THEN** 系统 MUST 允许全部模型参数参与训练
- **AND** 训练日志 MUST 记录 trainable parameter ratio 为 100% 或等价全量比例

### Requirement: 无标签与半监督 target adaptation
系统 MUST 支持 label budget 为 0 的无标签 target adaptation，以及 label budget 大于 0 时 labeled target loss 与 unlabeled target loss 的组合。无标签 loss MUST 只使用 `target_adapt` split，不得读取 `target_test` 标签或样本。

#### Scenario: 0-label adaptation 使用无标签目标
- **WHEN** `label_budget=0` 且配置启用 prototype alignment
- **THEN** 系统 MUST 使用 entropy minimization 和高置信 prototype consistency 进行 adaptation
- **AND** 系统 MUST 不读取 target label 作为 supervised loss

#### Scenario: Few-shot adaptation 合成监督与无监督 loss
- **WHEN** `label_budget` 大于 0 且 target_adapt 中存在未标注样本
- **THEN** 系统 MUST 对 labeled subset 计算 supervised hierarchical/flat loss
- **AND** 系统 MUST 按配置权重对 unlabeled subset 计算无监督 loss

#### Scenario: 低置信 prototype 样本被忽略
- **WHEN** target 样本到 source prototype 的最大 soft assignment 低于 confidence threshold
- **THEN** 系统 MUST 不对该样本计算 prototype consistency loss
- **AND** adaptation metrics MUST 记录 prototype coverage 或等价可诊断统计

### Requirement: HiST-Beam 指标与预测产物
HiST-Beam evaluation 和 adaptation MUST 输出 Top-1、Top-3、Top-5、coarse group accuracy、fine offset accuracy、trainable parameter ratio 和 adaptation time。若样本提供 beam power vector，系统 MUST 输出 normalized received power 和 beam power loss dB；若没有 power vector，系统 MUST 明确跳过 power 指标。

#### Scenario: 输出 coarse 和 fine 指标
- **WHEN** 评估 HiST-Beam hierarchical 变体
- **THEN** metrics MUST 包含 coarse group accuracy
- **AND** metrics MUST 包含 fine offset accuracy 或在 flat 变体中明确标记该指标不可用

#### Scenario: 输出 adaptation 效率指标
- **WHEN** target adaptation 完成
- **THEN** metrics MUST 包含 trainable parameter count、total parameter count、trainable parameter ratio 和 adaptation time
- **AND** adapter 变体的 trainable parameter ratio MUST 可与 full fine-tuning baseline 横向比较

#### Scenario: 保存 test predictions
- **WHEN** source-only evaluation 或 target adaptation evaluation 完成
- **THEN** 系统 MUST 保存 target_test predictions 文件
- **AND** predictions MUST 至少包含 sample id、scene、true beam、predicted beam、top-k predictions、coarse true/pred 和当前变体 metadata

#### Scenario: 缺失 beam power 时不伪造指标
- **WHEN** target_test 样本不包含 beam power vector
- **THEN** 系统 MUST 不输出虚假的 power gain 或 power loss 指标
- **AND** metrics MUST 记录 power metrics unavailable 的原因

