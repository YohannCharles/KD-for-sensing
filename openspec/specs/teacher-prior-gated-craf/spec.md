# teacher-prior-gated-craf Specification

## Purpose
TBD - created by archiving change add-teacher-prior-gated-craf. Update Purpose after archive.
## Requirements
### Requirement: Teacher reliability registry
系统 MUST 为 teacher-prior CRAF 提供可机器读取的 teacher reliability registry。registry MUST 记录场景、每个模态的 best checkpoint、验证指标、prior 和 prior 来源，并支持 `manual` 与 `metric` 两种 prior 模式。

#### Scenario: 构建手动 prior registry
- **WHEN** 用户运行 teacher registry 构建脚本并设置 `prior_mode: manual`
- **THEN** 系统 MUST 读取每个启用模态的 teacher `metrics.json` 和 best checkpoint
- **AND** 系统 MUST 将配置中的手动 prior 写入 registry
- **AND** registry MUST 包含 `scene`、`teachers.<modality>.ckpt`、`val_acc_top1`、`val_acc_top3`、`val_adba`、`prior` 和 `prior_mode`

#### Scenario: 构建 metric prior registry
- **WHEN** 用户运行 teacher registry 构建脚本并设置 `prior_mode: metric`
- **THEN** 系统 MUST 使用 teacher 验证指标计算每个模态的归一化 prior
- **AND** prior MUST 被 clamp 到配置的 `prior_min` 和 `prior_max`
- **AND** 系统 MUST 在 registry 中记录用于计算 prior 的指标权重

#### Scenario: teacher registry 缺少必需指标
- **WHEN** 某个 teacher 的 metrics 文件缺少 `val_acc_top1`、`val_acc_top3` 或 `val_adba`
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含缺失模态和缺失字段名

### Requirement: Prior residual gate
CRAF MUST 支持 `prior_residual_sigmoid` gate。该 gate MUST 使用 teacher registry 中的 prior logit 作为基线，加上可学习 residual logit 后计算 sigmoid gate，并 MUST 支持可用模态 mask、`min_gate` 和 diagnostics 输出。

#### Scenario: residual 零初始化后 gate 接近 prior
- **WHEN** 使用 prior `{image: 0.20, radar: 0.20, gps: 0.85, lidar: 0.15, mmwave: 0.90}` 初始化 `prior_residual_sigmoid`
- **THEN** 未训练 forward 的每模态 gate 均值 MUST 接近对应 prior
- **AND** 每个模态的绝对误差 MUST 小于 `0.03`

#### Scenario: 不可用模态 gate 清零
- **WHEN** forward 传入 `modality_mask=False` 的模态
- **THEN** 该模态输出 gate MUST 为 0
- **AND** prior regularization MUST 不在该模态位置产生 loss

#### Scenario: gate diagnostics 输出完整
- **WHEN** CRAF 使用 `prior_residual_sigmoid` 完成 forward
- **THEN** diagnostics MUST 包含 `gate`、`gate_logits`、`prior` 和 `residual_logits`
- **AND** diagnostics MUST 能按模态顺序映射到 `image`、`radar`、`gps`、`lidar` 和 `mmwave`

### Requirement: Teacher encoder initialization
Stage 2 CRAF MUST 能从 teacher registry 加载启用模态 teacher encoder 权重。加载过程 MUST 支持明确 key mapping、shape 校验、严格与非严格模式，并 MUST 将每模态加载结果写入训练日志。

#### Scenario: Stage 2 加载并冻结 teacher encoder
- **WHEN** Stage 2 配置设置 `teacher.load_encoders: true` 和 `teacher.freeze_encoders: true`
- **THEN** 系统 MUST 从 teacher registry 加载每个启用模态的 encoder 权重到 CRAF
- **AND** 每个成功加载的 encoder 参数 MUST 设置为 `requires_grad=False`
- **AND** fusion transformer、prediction head 和 prior residual gate 参数 MUST 保持 `requires_grad=True`

#### Scenario: teacher key 不匹配可诊断
- **WHEN** teacher checkpoint 中存在不能映射到 CRAF encoder 的 key 或 shape 不一致的 tensor
- **THEN** 系统 MUST 在 load summary 中记录 missing、unexpected 或 shape mismatch
- **AND** 严格模式下系统 MUST 拒绝继续训练

#### Scenario: 只加载配置指定模态
- **WHEN** Stage 2 配置只启用 GPS 和 mmWave
- **THEN** 系统 MUST 只尝试加载 GPS 和 mmWave teacher encoder
- **AND** 系统 MUST 不要求 image、radar 或 LiDAR teacher registry 项存在

### Requirement: Stage 2 prior-guided fusion training
Stage 2 CRAF 训练 MUST 默认冻结 encoder，并默认只优化 fusion transformer、prediction head、prior residual gate、task loss、beam soft 小权重和 prior regularization。Counterfactual、unimodal auxiliary 和 KD MUST 默认为关闭。

#### Scenario: Stage 2 默认 loss 组合
- **WHEN** 用户运行 Stage 2 teacher-init prior 配置
- **THEN** 总 loss MUST 包含 task loss、配置权重的 beam soft loss 和 prior regularization loss
- **AND** counterfactual loss、unimodal auxiliary loss 和 KD loss 的有效权重 MUST 为 0

#### Scenario: prior regularization 使用 mask
- **WHEN** 训练 batch 中某个模态被 modality dropout 或 force mask 置为不可用
- **THEN** prior regularization MUST 忽略该模态位置
- **AND** loss MUST 只在可用模态 gate 与 prior 之间计算

### Requirement: Stage 3 selective fine-tuning
Stage 3 CRAF MUST 从 Stage 2 best checkpoint 加载权重，并 MUST 根据配置选择性解冻强模态 encoder。默认配置 MUST 解冻 GPS 和 mmWave encoder，冻结 image、LiDAR 和 radar encoder。

#### Scenario: Stage 3 默认解冻强模态
- **WHEN** Stage 3 配置使用默认 `unfreeze_modalities: ["gps", "mmwave"]`
- **THEN** GPS 和 mmWave encoder 参数 MUST 设置为 `requires_grad=True`
- **AND** image、LiDAR 和 radar encoder 参数 MUST 设置为 `requires_grad=False`
- **AND** fusion transformer、prediction head 和 gate 参数 MUST 设置为 `requires_grad=True`

#### Scenario: Stage 3 参数组学习率
- **WHEN** 系统为 Stage 3 构建 optimizer
- **THEN** fusion、head、gate、strong encoder 和 weak encoder 参数组 MUST 能使用不同学习率
- **AND** 默认 strong encoder 学习率 MUST 小于 fusion/head/gate 学习率

### Requirement: Optional reliability KD and counterfactual ablations
系统 MUST 保留 reliability-weighted KD、relative context marginal counterfactual 和 shuffle counterfactual 作为显式 ablation。它们 MUST 默认关闭，并且不得改变 Stage 2/3 主实验默认行为。

#### Scenario: reliability-weighted KD 默认关闭
- **WHEN** Stage 2 或 Stage 3 主实验配置未显式启用 KD
- **THEN** 系统 MUST 不构建 teacher logits KD loss
- **AND** 训练日志中的 KD loss 有效权重 MUST 为 0

#### Scenario: 只使用强 teacher 的 KD ablation
- **WHEN** 用户显式设置 `kd.enabled: true` 且 `kd.use_modalities: ["gps", "mmwave"]`
- **THEN** reliability-weighted KD MUST 只使用 GPS 和 mmWave teacher logits
- **AND** 系统 MUST 不把 image、radar 或 LiDAR teacher logits 纳入 KD 权重计算

#### Scenario: shuffle counterfactual 只用 CE delta
- **WHEN** 用户显式启用 `counterfactual.mode: shuffle`
- **THEN** counterfactual delta MUST 只使用 per-sample CE 或 label-smoothed CE
- **AND** delta MUST 不混入 beam soft、unimodal auxiliary、KD、prior regularization 或 gate loss

### Requirement: Teacher-prior CRAF diagnostics
系统 MUST 为 teacher-prior CRAF 输出可诊断日志。日志 MUST 覆盖 gate、prior、residual、teacher load/freeze 状态、分项 loss 和模态组合评估。

#### Scenario: 记录 gate 与 prior 指标
- **WHEN** teacher-prior CRAF 完成一个训练 epoch
- **THEN** 训练日志和 TensorBoard MUST 记录每模态 gate 均值
- **AND** 训练日志和 TensorBoard MUST 记录每模态 prior 与 residual logit 均值

#### Scenario: 记录 teacher 加载冻结状态
- **WHEN** Stage 2 或 Stage 3 初始化完成
- **THEN** 训练日志 MUST 记录每模态 teacher load success、load summary、frozen 状态和 trainable parameter count

#### Scenario: 记录模态组合验证
- **WHEN** 配置启用 teacher-prior CRAF 模态组合评估
- **THEN** 验证输出 MUST 包含 `gps`、`mmwave`、`gps_mmwave`、`strong_only`、`weak_only` 和 `all` 的指标
- **AND** 每个组合 MUST 至少包含 Top-1、ATop-3、ATop-5、ADBA 和 loss

