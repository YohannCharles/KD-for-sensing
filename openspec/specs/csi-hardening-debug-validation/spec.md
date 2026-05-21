# csi-hardening-debug-validation Specification

## Purpose
TBD - created by archiving change debug-csi-hardening-experiment-matrix. Update Purpose after archive.
## Requirements
### Requirement: CSI hardening debug run matrix
系统 MUST 提供一个最小 CSI hardening debug run 矩阵，用于先验证配置、数据流和训练路径，再解释完整 sweep 结果。该矩阵 MUST 包含 `A0_original`、`A0_clone_generated`、`A0_clone_generated + pilot disabled through new path`、`C1_view_gate_warmup_only` 和 `C2_no_internal_gru_only`。

#### Scenario: 生成最小 debug 矩阵
- **WHEN** 开发者请求 CSI hardening debug 矩阵
- **THEN** 系统 MUST 生成或提供上述 5 个 run 的配置入口
- **AND** 每个 debug run MUST 支持设置 10 到 20 epoch 的短训练参数

#### Scenario: 暂停完整 sweep 结论
- **WHEN** `A0_clone_generated` 尚未证明接近 `A0_original`
- **THEN** 系统 MUST 将完整 A/B/C/D sweep 结果标记为待排查
- **AND** 分析输出 MUST 不把非 A0 变体的低 accuracy 解释为 hardening 设计失败

### Requirement: Generated A0 parity gate
系统 MUST 支持通过当前 sweep/config 生成器生成 `A0_clone_generated`，并将其关键配置与 `A0_original` 对齐。除 run identity、输出目录、seed 或时间戳字段外，关键训练、数据、模型和 CSI encoder 字段 MUST 一致。

#### Scenario: A0 clone 关闭全部新选项
- **WHEN** 系统生成 `A0_clone_generated`
- **THEN** 配置 MUST 关闭 `csi_hardening`
- **AND** 配置 MUST 关闭 `csi_degradation`
- **AND** 配置 MUST 关闭 pilot noise
- **AND** 配置 MUST 设置 `use_internal_gru=true`、`view_fusion=symmetric_gate`、`view_gate_warmup_epochs=0` 和 `delay_view_warmup_epochs=0`
- **AND** 配置 MUST 使用与 A0 original 相同的 `representation_core`

#### Scenario: A0 config diff 发现关键差异
- **WHEN** 系统比较 `A0_original` 和 `A0_clone_generated` 的 resolved config
- **THEN** diff artifact MUST 标出 optimizer、scheduler、loss、dataset split、normalization、train RMS path、`seq_len`、`num_pred`、`num_classes`、CSI encoder、representation core 和 beam head 的差异
- **AND** 如果任一关键字段不同，debug 状态 MUST 标记为配置继承失败

### Requirement: CSI batch dataflow diagnostics
系统 MUST 在 CSI debug run 的第一个 train batch 和第一个 val batch 记录 CSI 数据流统计。统计 MUST 覆盖 hardening 前、hardening 后、pilot 后、freq view、delay view、view features、fused feature、GRU 输出和最终 CSI feature。

#### Scenario: 记录 complex CSI 张量统计
- **WHEN** debug logging 启用且 CSI encoder 处理首个 train 或 val batch
- **THEN** 系统 MUST 记录 hardening 前、hardening 后和 pilot 后 CSI 的 shape、dtype、abs_mean、abs_std、abs_max、real_mean、imag_mean、nan_count 和 zero_ratio
- **AND** 记录的 hardening 后和 pilot 后 shape MUST 可用于检查 `[B,T,Nsc,Nant]` 契约

#### Scenario: 记录 view 与 feature norm
- **WHEN** debug logging 启用且 CSI encoder 生成双视图特征
- **THEN** 系统 MUST 记录 freq view 和 delay view 的 shape、mean、std 和 nan_count
- **AND** 系统 MUST 记录 freq_feat、delay_feat、fused_feat、gru_out 和 final CSI feature 的 norm

### Requirement: Debug decision rules
系统 MUST 为最小 debug 矩阵提供判定规则，使排查能区分配置继承、view warmup、no internal GRU、hardening 和 pilot estimator 错误。

#### Scenario: A0 clone 低于 A0 original
- **WHEN** `A0_original` 正常学习而 `A0_clone_generated` 掉到接近随机水平
- **THEN** debug 结论 MUST 指向配置生成或继承错误
- **AND** 系统 MUST 阻止把该结果解释为 hardening 效果

#### Scenario: C1 或 C2 单独崩溃
- **WHEN** `A0_clone_generated` 正常学习但 `C1_view_gate_warmup_only` 或 `C2_no_internal_gru_only` 掉到接近随机水平
- **THEN** debug 结论 MUST 分别指向 view gate warmup 实现或 no-internal-GRU 路径错误

#### Scenario: Pilot disabled path 异常
- **WHEN** `A0_clone_generated + pilot disabled through new path` 与 `A0_clone_generated` 明显不一致
- **THEN** debug 结论 MUST 指向 pilot estimator enabled/disabled 配置解析或数据流错误

### Requirement: Training health diagnostics
系统 MUST 在 CSI debug run 中按 epoch 记录关键模块的梯度范数和参数变化，用于确认模型真的接到 loss 并被 optimizer 更新。

#### Scenario: 记录模块梯度和参数变化
- **WHEN** debug training health logging 启用且一个训练 epoch 完成
- **THEN** 系统 MUST 记录 `grad_norm_csi_encoder`、`grad_norm_representation_core` 和 `grad_norm_beam_head`
- **AND** 系统 MUST 记录 `param_delta_csi_encoder`、`param_delta_representation_core` 和 `param_delta_beam_head`

#### Scenario: 发现模块未训练
- **WHEN** 任一关键模块的 grad norm 或 param delta 持续为 0
- **THEN** debug 输出 MUST 标记该模块可能被冻结、未加入 optimizer、被梯度屏蔽或未连接到 loss

