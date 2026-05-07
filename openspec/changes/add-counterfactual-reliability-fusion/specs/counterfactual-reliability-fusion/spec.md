## ADDED Requirements

### Requirement: CRAF fusion 模型构建
系统 MUST 提供可通过配置构建的 CRAF fusion 模型。该模型 MUST 支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意非空有效组合，并 MUST 复用项目固定模态顺序和模态标准化规则。

#### Scenario: 构建五模态 CRAF
- **WHEN** 用户配置 CRAF 模型并设置 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 构建五个模态分支或 projector
- **AND** 模型 MUST 接收现有 fusion 输入键对应的五个张量
- **AND** 模型 MUST 输出 beam logits

#### Scenario: 构建任意双模态 CRAF
- **WHEN** 用户配置 CRAF 模型并设置任意合法双模态组合
- **THEN** 系统 MUST 只要求该组合对应的输入张量
- **AND** 系统 MUST 不要求未启用模态字段存在

#### Scenario: 拒绝非法 CRAF 模态
- **WHEN** 用户配置 CRAF 模型并包含未知模态、重复模态或空模态列表
- **THEN** 系统 MUST 拒绝构建模型
- **AND** 错误信息 MUST 包含非法模态或可用模态列表

### Requirement: Token 化与模态 mask
CRAF 模型 MUST 将每个启用模态的时序输入转换为统一维度 token，并 MUST 生成 attention 可用的 token padding mask。不可用或被强制 drop 的模态 token MUST 在 attention 中被忽略。

#### Scenario: 默认所有启用模态可用
- **WHEN** batch 未提供真实 `modality_mask` 且训练未传入 `force_modality_mask`
- **THEN** CRAF MUST 将配置启用的所有模态视为可用
- **AND** token padding mask MUST 不屏蔽这些模态的历史 token

#### Scenario: 强制 drop 部分模态
- **WHEN** 训练 helper 传入 `force_modality_mask` 屏蔽某个启用模态
- **THEN** CRAF MUST 将该模态 token 标记为 padding 或等价忽略状态
- **AND** 该模态的 reliability gate MUST 为 0 或不参与融合贡献

#### Scenario: token 形状稳定
- **WHEN** batch size 为 `B`、启用模态数为 `K`、历史长度为 `T`
- **THEN** tokenizer 输出 token MUST 能表示为 `[B, K, T, D]`
- **AND** token padding mask MUST 能表示为 `[B, K, T]`

### Requirement: Reliability 估计与 gate
CRAF MUST 为每个样本、每个启用模态估计 reliability 分数，并 MUST 使用该分数门控对应模态 token。Reliability 分数 MUST 同时受模态表示、单模态预测 confidence 和可选 dataset prior 影响。

#### Scenario: 输出 reliability
- **WHEN** CRAF 完成一次 forward
- **THEN** 输出 MUST 包含每样本每模态的 reliability 分数
- **AND** reliability 张量最后一维 MUST 与启用模态顺序一致

#### Scenario: 屏蔽不可用模态 reliability
- **WHEN** 某个模态在 effective modality mask 中不可用
- **THEN** 该模态 reliability MUST 不参与最终融合
- **AND** 输出的 effective modality mask MUST 能用于解释该行为

#### Scenario: 防止 gate 早期完全压死 token
- **WHEN** 配置设置 `min_gate` 大于 0
- **THEN** 可用模态 token 的有效 gate MUST 不低于 `min_gate`
- **AND** 不可用模态仍 MUST 被 mask 忽略

### Requirement: 单模态辅助预测与 confidence
CRAF MUST 为每个启用模态提供轻量单模态辅助预测头，并 MUST 从辅助预测中计算可用于 reliability 估计的 confidence 特征。

#### Scenario: 单模态 logits 输出
- **WHEN** CRAF forward 启用 `return_unimodal`
- **THEN** 输出 MUST 包含单模态 logits
- **AND** 单模态 logits MUST 能按 `[B, K, H, C]` 或等价结构表示

#### Scenario: confidence 特征计算
- **WHEN** 单模态 logits 可用
- **THEN** 系统 MUST 计算 entropy-based confidence 和 top probability margin
- **AND** confidence 特征 MUST 与启用模态顺序一致

#### Scenario: 单模态辅助 loss 可关闭
- **WHEN** 配置将单模态辅助 loss 权重设为 0
- **THEN** 训练总 loss MUST 不包含单模态辅助 loss
- **AND** forward 输出仍 MAY 保留 diagnostics 字段

### Requirement: Transformer fusion 与 horizon prediction
CRAF MUST 使用 token-level fusion 模块融合启用模态历史 token，并 MUST 输出与当前训练标签语义一致的预测 slot。

#### Scenario: Transformer 忽略 padding token
- **WHEN** token padding mask 中某些位置为 True
- **THEN** Transformer fusion MUST 在 self-attention 中忽略这些 token

#### Scenario: 预测长度对齐现有标签
- **WHEN** 配置中的 `model.num_pred` 为 `N`
- **THEN** CRAF 默认 MUST 输出 `N + 1` 个预测 slot
- **AND** 这些 slot MUST 能直接与 `prepare_labels()` 的输出对齐

#### Scenario: 输出类别数对齐
- **WHEN** 配置中的 `model.num_classes` 为 `C`
- **THEN** CRAF 输出 logits 的最后一维 MUST 为 `C`

### Requirement: Counterfactual gate supervision
训练流程 MUST 能在 CRAF 显式启用时执行反事实 forward，以估计某个模态被移除后的性能变化，并用该贡献目标监督 reliability gate。

#### Scenario: 反事实训练关闭
- **WHEN** `training.counterfactual.enabled` 为 false 或缺省
- **THEN** 训练流程 MUST 只执行普通 forward 和普通 loss
- **AND** 既有非 CRAF 模型训练行为 MUST 不变

#### Scenario: sample-one 反事实
- **WHEN** `training.counterfactual.mode` 为 `sample_one`
- **THEN** 每个 batch MUST 至少采样一个可用模态执行 drop-forward
- **AND** 训练流程 MUST 基于 full-forward 与 drop-forward 的 per-sample loss 差异形成 gate target

#### Scenario: leave-one-out 反事实
- **WHEN** `training.counterfactual.mode` 为 `leave_one_out`
- **THEN** 训练流程 MUST 对可用模态逐一构造 drop mask
- **AND** 训练流程 MUST 汇总各模态贡献目标

#### Scenario: 反事实 warmup
- **WHEN** 当前 epoch 小于 `training.counterfactual.start_epoch`
- **THEN** 训练流程 MUST 跳过 counterfactual gate supervision
- **AND** 仍 MUST 执行普通任务训练

### Requirement: Beam-aware soft label loss
系统 MUST 提供 beam-aware soft label loss，使邻近 beam 的预测误差小于远离 beam 的预测误差，并 MUST 支持关闭该 loss。

#### Scenario: 构造 beam soft target
- **WHEN** labels、beam 数量和 sigma 可用
- **THEN** 系统 MUST 为每个有效标签构造 beam 距离衰减的 soft target
- **AND** ignore index 位置 MUST 不参与 loss

#### Scenario: circular beam 距离
- **WHEN** 配置启用 circular beam 距离
- **THEN** soft target 的距离计算 MUST 使用环形类别距离

#### Scenario: soft loss 权重为 0
- **WHEN** `loss.beam_soft.weight` 为 0 或 soft loss 未启用
- **THEN** 训练总 loss MUST 不包含 beam-aware soft label loss

### Requirement: CRAF diagnostics 输出
CRAF 训练和评估 MUST 能记录可靠性相关诊断信息，以便分析模态失衡和 gate 行为。

#### Scenario: 训练日志包含 reliability 摘要
- **WHEN** CRAF 训练完成一个 epoch
- **THEN** 运行日志 MUST 能记录各启用模态的平均 reliability
- **AND** 日志 MUST 能记录 counterfactual loss 和单模态辅助 loss 的 epoch 摘要

#### Scenario: TensorBoard 可选记录 reliability
- **WHEN** TensorBoard 启用且 CRAF 输出 reliability diagnostics
- **THEN** 系统 MUST 写入每个启用模态的 reliability 标量或等价摘要

#### Scenario: 非 CRAF 模型无 diagnostics
- **WHEN** 训练模型不是 CRAF 或未返回 diagnostics
- **THEN** 日志写入 MUST 跳过 CRAF 专属字段
- **AND** 训练和验证 MUST 不报错

### Requirement: CRAF baseline 对比
项目 MUST 提供与 CRAF 可比较的 baseline 模型或配置，用于区分 token 化、Transformer 和 reliability gate 的贡献。

#### Scenario: token-only transformer baseline
- **WHEN** 用户选择 token transformer fusion baseline
- **THEN** 模型 MUST 使用与 CRAF 相同的 token 化和 Transformer fusion
- **AND** 模型 MUST 不使用 reliability gate

#### Scenario: early concat baseline
- **WHEN** 用户选择 early concat GRU 或 early concat Transformer baseline
- **THEN** 系统 MUST 使用启用模态的帧级特征拼接作为融合输入
- **AND** 该 baseline MUST 能复用现有 fusion batch 输入

#### Scenario: single modal transformer baseline
- **WHEN** 用户选择单模态 transformer baseline
- **THEN** 模型 MUST 只消费一个模态输入
- **AND** 输出 MUST 与同一训练/评估流程的标签语义对齐
