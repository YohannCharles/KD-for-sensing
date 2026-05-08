## ADDED Requirements

### Requirement: MARF fusion model construction
系统 MUST 提供可通过 registry 构建的 MARF fusion student。该模型 MUST 使用项目固定模态顺序，支持 `image`、`radar`、`gps`、`lidar`、`mmwave` 的任意非空有效组合，并 MUST 保持现有 `experiment.task: fusion` 输入契约。

#### Scenario: 构建五模态 MARF
- **WHEN** 用户配置 `model.student.type: marf_fusion` 且 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 构建五个模态 encoder 或 projector
- **AND** 模型 MUST 接收现有 fusion 输入键对应的五个张量
- **AND** 模型 MUST 输出 beam logits

#### Scenario: 构建任意子集 MARF
- **WHEN** 用户配置 `marf_fusion` 且 `modalities` 为任意合法非空模态组合
- **THEN** 模型 MUST 只要求该组合对应的输入张量
- **AND** 输出 diagnostics 中的模态顺序 MUST 与标准化后的 `modalities` 一致

#### Scenario: MARF horizon 对齐现有标签
- **WHEN** 配置中的 `model.num_pred` 为 `N`
- **THEN** MARF logits MUST 输出 `N + 1` 个 prediction slot
- **AND** 这些 slot MUST 能直接传入现有 `select_prediction_slots()` 与 `prepare_labels()` 结果对齐

### Requirement: MARF forward output contract
MARF forward MUST 返回包含主 logits 与路由诊断的 dict。输出张量 MUST 能支持训练、验证、subset 评估和 TensorBoard 诊断。

#### Scenario: 输出核心张量
- **WHEN** batch size 为 `B`、启用模态数为 `K`、历史长度为 `T`、hidden 维度为 `D`、预测 slot 数为 `H`、beam 类别数为 `C`
- **THEN** `outputs["logits"]` MUST 具有形状 `[B, H, C]`
- **AND** `outputs["token_features"]` MUST 能表示为 `[B, K, T, D]`
- **AND** `outputs["anchor_weights"]` 和 `outputs["residual_weights"]` MUST 具有形状 `[B, H, K]`
- **AND** `outputs["h_anchor"]` 和 `outputs["h_final"]` MUST 具有形状 `[B, H, D]`

#### Scenario: 输出路由诊断
- **WHEN** MARF 完成 forward
- **THEN** 输出 MUST 包含 `anchor_logits`、`anchor_weights`、`residual_logits`、`residual_weights`、`residual_delta`、`effective_modality_mask`、`prior` 和 `modalities`
- **AND** `adapt_model_output()` MUST 能从该 dict 中解析 logits、input features、output features 和 diagnostics

### Requirement: Horizon-wise modality router
MARF router MUST 为每个样本、每个 prediction slot、每个启用模态生成 anchor 与 residual 路由权重。Router MUST 不写死任何模态为强模态或弱模态。

#### Scenario: Anchor 权重归一化
- **WHEN** 每个样本至少有一个可用模态
- **THEN** `anchor_weights` MUST 在每个样本和每个 prediction slot 的可用模态集合上通过 softmax 归一化
- **AND** 每个 `[B,H]` 位置的可用模态 anchor 权重和 MUST 接近 1

#### Scenario: 不可用模态被 mask
- **WHEN** forward 传入 `force_modality_mask` 屏蔽某个模态
- **THEN** 该模态在所有 prediction slot 的 `anchor_weights` MUST 为 0
- **AND** 该模态在所有 prediction slot 的 `residual_weights` MUST 为 0
- **AND** 该模态 token MUST 不参与 anchor attention 或 residual attention 的有效贡献

#### Scenario: Prior bias 可关闭
- **WHEN** 配置设置 `model.student.router.use_prior_bias: false`
- **THEN** router MUST 不把 teacher prior logit 加入 anchor 或 residual logits
- **AND** forward 仍 MUST 输出用于诊断的 prior 张量

#### Scenario: Prior bias 不是固定规则
- **WHEN** 配置启用 prior bias
- **THEN** router MAY 将 teacher prior logit 作为 anchor/residual logits 的可缩放 bias
- **AND** 所有启用模态仍 MUST 能在不同样本或不同 prediction slot 获得非零 anchor 或 residual 权重

### Requirement: Anchor fusion and residual adapter
MARF MUST 使用 horizon-wise anchor fusion 生成主表示，并使用 conditional residual adapter 允许任意模态提供补充信息。

#### Scenario: Anchor fusion 使用 horizon query
- **WHEN** MARF 执行 anchor fusion
- **THEN** 每个 prediction slot MUST 使用独立或等价的 learnable query 从加权模态 token 中生成 `h_anchor`
- **AND** token padding mask MUST 阻止被屏蔽模态 token 参与 attention

#### Scenario: Residual adapter 汇总每模态 delta
- **WHEN** residual adapter 启用
- **THEN** 每个启用模态 MUST 能基于 `h_anchor` 和自身 token 产生 `residual_delta`
- **AND** `h_final` MUST 由 `h_anchor` 加上按 `residual_weights` 加权后的 residual 汇总得到

#### Scenario: No-residual ablation
- **WHEN** 配置设置 `model.student.residual_adapter.enabled: false`
- **THEN** MARF MUST 跳过 residual adapter 对 `h_final` 的影响
- **AND** 模型仍 MUST 输出 logits、anchor weights 和可用的 diagnostics

### Requirement: MARF subset-aware training
训练流程 MUST 在 MARF 显式启用 subset-aware training 时，对 all-modal forward 和若干 prior-driven subset forward 联合优化。

#### Scenario: Subset training 关闭
- **WHEN** `training.subset_training.enabled` 为 false 或缺省
- **THEN** 训练流程 MUST 只执行普通 forward 和普通 loss
- **AND** 既有 CRAF、token transformer 和 legacy fusion 训练行为 MUST 不变

#### Scenario: Subset training 前向
- **WHEN** MARF 配置启用 `training.subset_training.enabled`
- **THEN** 每个 batch MUST 先执行 all-modal forward
- **AND** 训练流程 MUST 按配置采样 `top_prior`、`random_with_top_prior` 或其它 subset mask 并执行 subset forward
- **AND** subset forward MUST 通过 `force_modality_mask` 控制可用模态

#### Scenario: Subset loss 组合
- **WHEN** subset-aware training 产生 subset forward
- **THEN** 总 loss MUST 能包含 all-modal task loss、subset CE、all-to-subset KD、beam soft loss、residual norm loss、prior regularization 和可选 anchor entropy loss
- **AND** all-to-subset KD MUST 使用 all-modal logits 的 detached soft target

### Requirement: Prior-driven subset sampler
系统 MUST 提供 prior-driven modality subset sampler。Sampler MUST 从 teacher prior 或等价配置中决定 top-prior 模态，不得写死 GPS/mmWave。

#### Scenario: 采样 top-prior subset
- **WHEN** sampler 收到每模态 prior 和 `top_prior_k`
- **THEN** `top_prior` 模式 MUST 选择 prior 最高的可用模态集合
- **AND** 选择结果 MUST 随输入 prior 改变

#### Scenario: 采样 random-with-top-prior subset
- **WHEN** sampler 使用 `random_with_top_prior` 模式
- **THEN** 每个样本的 subset MUST 至少包含一个当前可用的 top-prior 模态
- **AND** subset MUST 满足配置的最小保留模态数

#### Scenario: 采样 drop-one subset
- **WHEN** sampler 使用 `drop_one` 模式且样本有两个以上可用模态
- **THEN** sampler MUST 从可用模态中随机删除一个模态
- **AND** 输出 mask MUST 至少保留配置允许的最小模态数

### Requirement: MARF diagnostics and logging
训练和验证流程 MUST 能记录 MARF router、residual 和 subset training 的诊断标量。

#### Scenario: 记录 router 权重
- **WHEN** MARF 完成一个训练 epoch
- **THEN** 训练日志和 TensorBoard MUST 能记录每模态 anchor 均值和 residual 均值
- **AND** 训练日志和 TensorBoard MUST 能记录每个 prediction slot 的每模态 anchor 均值

#### Scenario: 记录 subset loss
- **WHEN** subset-aware training 启用
- **THEN** 训练日志 MUST 包含 all loss、subset CE、subset KD、beam soft、residual norm、prior regularization 和 anchor entropy 的有效标量或等价字段
- **AND** 权重为 0 的 loss MUST 记录为 0 或跳过且不得影响总 loss

### Requirement: MARF evaluation scripts
项目 MUST 提供用于验证 MARF 结论可信性的调试和评估脚本。脚本 MUST 使用现有配置加载、checkpoint 加载、dataloader 和 validator 语义。

#### Scenario: Evaluation consistency debug
- **WHEN** 用户运行 MARF eval consistency 脚本并提供配置和 checkpoint
- **THEN** 脚本 MUST 输出 official validation Top-1、subset all Top-1、二者差值、样本数、batch 数、logits 形状和 labels 形状
- **AND** 当差值大于配置阈值时脚本 MUST 以非零状态或清晰错误报告失败

#### Scenario: Modality perturbation evaluation
- **WHEN** 用户运行 perturbation 脚本并指定 `shuffle_<modality>` 或 `zero_<modality>`
- **THEN** 脚本 MUST 在同一 checkpoint 上输出 clean 指标和扰动后指标
- **AND** 脚本 MUST 不改变训练集 normalization artifact 复用语义

#### Scenario: Modality subsets evaluation
- **WHEN** 用户运行 subset evaluation 脚本
- **THEN** 脚本 MUST 支持 `all`、`top_prior`、`single_best_prior`、`random_with_top_prior` 和 low-prior subset
- **AND** 输出 MUST 记录每个 subset 实际使用的模态列表

### Requirement: MARF experiment configs and tests
项目 MUST 提供 MARF 主实验、subset training 和必要 ablation 配置，并 MUST 提供覆盖核心行为的自动化测试。

#### Scenario: MARF 主配置可加载
- **WHEN** 用户加载 MARF Scene32 主配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 使用 `model.student.type: marf_fusion`
- **AND** 配置 MUST 提供 teacher registry 路径、encoder 加载/冻结策略、router、anchor fusion、residual adapter、loss 和 evaluation subset 字段

#### Scenario: MARF ablation 配置可加载
- **WHEN** 用户加载 no residual、no prior bias 或 no subset training ablation 配置
- **THEN** 每个配置 MUST 只改变对应 ablation 字段
- **AND** 每个配置 MUST 保持同一场景、同一 split、同一模态集合和同一基础训练超参数

#### Scenario: MARF 单元测试
- **WHEN** 开发者运行 MARF 定向测试
- **THEN** 测试 MUST 覆盖 forward shape、anchor softmax、mask 清零、prior bias 开关和 no-residual ablation
- **AND** 测试命令 MUST 使用 `conda run -n kd_mm_beam` 环境约束
