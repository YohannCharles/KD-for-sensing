# counterfactual-reliability-fusion Specification

## Purpose
TBD - created by archiving change add-counterfactual-reliability-fusion. Update Purpose after archive.
## Requirements
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

### Requirement: 分阶段 reliability gate 训练
CRAF 训练 MUST 支持 warmup 阶段固定可用模态 gate，并 MUST 在配置指定的 counterfactual 起始 epoch 后才启用反事实 gate supervision。

#### Scenario: warmup 阶段固定 gate
- **WHEN** 当前 epoch 小于 `training.counterfactual.start_epoch` 或 `training.warmup_epochs`
- **THEN** CRAF MUST 对所有可用模态使用等价于 `r_m = 1` 的有效 gate
- **AND** 训练流程 MUST 跳过 counterfactual gate loss

#### Scenario: warmup 后启用 learned gate
- **WHEN** 当前 epoch 达到 counterfactual 起始 epoch 且 `training.counterfactual.enabled` 为 true
- **THEN** CRAF MUST 使用 reliability estimator 产生的 gate 门控可用模态 token
- **AND** 训练流程 MUST 允许 counterfactual gate supervision 参与总 loss

#### Scenario: gate loss 权重 ramp
- **WHEN** 配置设置 `loss.gate_ramp_epochs` 大于 0
- **THEN** 训练流程 MUST 从 counterfactual 起始 epoch 开始线性增加 gate loss 有效权重
- **AND** 有效权重 MUST 不超过配置的目标 `loss.gate_weight`

### Requirement: CE-only 反事实贡献目标
Counterfactual gate target MUST 基于主任务 sequence CE 的 per-sample loss 差异计算，并 MUST 支持对不明确贡献使用 ignore band。

#### Scenario: 贡献差值只使用 CE
- **WHEN** 训练流程计算 `delta` 用于监督 reliability gate
- **THEN** `delta` MUST 只由 full/context forward 与 counterfactual forward 的主任务 CE 差异产生
- **AND** `delta` MUST 不包含 beam soft loss、单模态 auxiliary loss、KD loss 或 gate loss

#### Scenario: ignore band 跳过模糊贡献
- **WHEN** `abs(delta)` 小于或等于 `training.counterfactual.ignore_delta_eps`
- **THEN** 该样本-模态 pair MUST 不参与 gate target loss
- **AND** 训练日志 MUST 能统计该模态的 target 有效率

#### Scenario: 二值 gate target
- **WHEN** `delta` 大于 `training.counterfactual.ignore_delta_eps`
- **THEN** gate target MUST 表示该模态有正贡献
- **AND** 当 `delta` 小于负的 `training.counterfactual.ignore_delta_eps` 时，gate target MUST 表示该模态有负贡献

### Requirement: context-marginal 反事实模式
训练流程 MUST 支持 `context_marginal` 反事实模式，用随机上下文子集估计某个模态的条件边际贡献。

#### Scenario: 构造不含目标模态的上下文
- **WHEN** `training.counterfactual.mode` 为 `context_marginal`
- **THEN** 训练流程 MUST 为目标模态采样一个不包含该模态的可用模态上下文子集 `A`
- **AND** `A` MUST 至少保留配置允许的最小模态数量

#### Scenario: 比较加入目标模态前后 CE
- **WHEN** 上下文子集 `A` 和目标模态 `m` 已确定
- **THEN** 训练流程 MUST 分别计算 `A` 与 `A ∪ {m}` 的主任务 CE
- **AND** 训练流程 MUST 使用 `CE(A) - CE(A ∪ {m})` 作为该模态的贡献差值

#### Scenario: 保留旧反事实模式
- **WHEN** 用户配置 `sample_one` 或 `leave_one_out`
- **THEN** 训练流程 MUST 继续支持已有 drop-forward 语义
- **AND** 新的 `context_marginal` 模式 MUST 不改变旧模式配置的行为

### Requirement: competitive modality gate
CRAF reliability gate MUST 支持在可用模态集合上进行 softmax 归一化，使模态之间形成竞争，并 MUST 保持不可用模态不参与融合。

#### Scenario: softmax gate 归一化
- **WHEN** `model.student.reliability.gate_type` 为 `softmax`
- **THEN** CRAF MUST 只在 effective modality mask 标记为可用的模态上计算 softmax gate
- **AND** 不可用模态的有效 gate MUST 为 0 或等价不参与融合

#### Scenario: gate 温度退火
- **WHEN** 配置提供 `gate_temperature_start` 和 `gate_temperature_end`
- **THEN** CRAF 训练 MUST 按 epoch 计算当前 gate temperature
- **AND** 温度 MUST 在配置边界内从起始值逐步过渡到结束值

#### Scenario: 保持 token 幅值尺度
- **WHEN** softmax gate 在 `K` 个可用模态上产生归一化权重
- **THEN** CRAF MUST 支持将 gate 按可用模态数缩放或使用等价方式保持融合 token 的整体幅值尺度
- **AND** `min_gate` MUST 只作用于可用模态

### Requirement: CRAF 附加 loss 调度
CRAF 训练 MUST 支持对单模态 auxiliary loss、beam soft loss 和 gate loss 进行配置化调度，避免附加目标在主任务未稳定时主导优化。

#### Scenario: 单模态 auxiliary warmup-only
- **WHEN** 配置设置 `loss.uni_weight_warmup` 大于 0 且 `loss.uni_weight_after_warmup` 为 0
- **THEN** 训练流程 MUST 只在 warmup 阶段将单模态 auxiliary loss 加入总 loss
- **AND** warmup 后总 loss MUST 不包含单模态 auxiliary loss

#### Scenario: 单模态 auxiliary 两段权重
- **WHEN** 配置同时提供 `loss.uni_weight_warmup` 和 `loss.uni_weight_after_warmup`
- **THEN** 训练流程 MUST 根据当前 epoch 选择对应阶段权重
- **AND** 日志 MUST 能记录实际生效的 auxiliary loss 权重

#### Scenario: beam soft loss 可降权
- **WHEN** 配置设置 `loss.beam_soft.weight`
- **THEN** 训练流程 MUST 按该权重加入 beam-aware soft label loss
- **AND** 权重为 0 时训练总 loss MUST 不包含 beam soft loss

### Requirement: CRAF 反事实诊断日志
CRAF 训练 MUST 记录足以判断 reliability supervision 是否有效的每模态 counterfactual 诊断标量。

#### Scenario: 记录每模态 delta 均值
- **WHEN** counterfactual supervision 在一个 epoch 内产生有效 `delta`
- **THEN** 训练日志 MUST 包含每个启用模态的 `cf/delta_mean_<modality>` 或等价字段

#### Scenario: 记录每模态 target 均值
- **WHEN** counterfactual supervision 在一个 epoch 内产生有效 gate target
- **THEN** 训练日志 MUST 包含每个启用模态的 `cf/target_mean_<modality>` 或等价字段

#### Scenario: 记录每模态 target 有效率
- **WHEN** 配置启用 `training.counterfactual.ignore_delta_eps`
- **THEN** 训练日志 MUST 包含每个启用模态的 `cf/target_valid_rate_<modality>` 或等价字段
- **AND** 该字段 MUST 反映未被 ignore band 跳过的样本-模态比例

