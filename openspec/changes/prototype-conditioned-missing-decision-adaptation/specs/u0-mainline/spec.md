## ADDED Requirements

### Requirement: 冻结 U0 的缺失模式决策适配实验
系统 MUST 支持一个实验性包装器，在不改变 U0 编码器、projection、融合、prototype、分类器、归一化统计、训练状态或原始 logits 的条件下，仅对非 Full 的四模态 mask 应用共享低秩决策残差。四模态顺序 MUST 为 `[image, radar, gps, lidar]`；Full 行 MUST 通过显式旁路直接返回 U0 logits，不得调用 Adapter 或依赖学习 gate。

#### Scenario: Full mask 的逐样本严格等价
- **WHEN** 已冻结 U0 的包装器接收 `[1,1,1,1]` 的任意 batch
- **THEN** 输出 logits MUST 等于同一 U0 的 base logits，最大绝对差不超过 `1e-7`，且 argmax 完全一致

#### Scenario: 非 Full mask 的零初始化
- **WHEN** Adapter 的输出矩阵为零初始化并接收任意非空非 Full mask
- **THEN** 新 logits MUST 等于 U0 对该 mask 的 base logits，且 U0 参数的 `requires_grad` 全部为 false

### Requirement: U0 prototype state 只读诊断
当 U0 使用 prototype head 时，系统 MUST 从同一组已训练 Beam prototype 导出 assignment、nearest id、nearest distance、distance margin、entropy 与 restoration residual norm 的只读诊断状态。该状态 MUST 不读取标签、不创建可训练状态、不更新 prototype，且在进入 Adapter 条件网络前 detach。

#### Scenario: 缺失条件 Adapter 使用 prototype state
- **WHEN** prototype-conditioned Adapter 对一个非 Full batch 前向
- **THEN** 条件输入 MUST 来自当前输入特征与既有 prototype，且其张量不携带到 U0/prototype 的梯度路径

### Requirement: A0--A7 的受控决策适配比较
Stage A MUST 包含 A0 冻结 U0、A1 mask bias、A2/A3/A4 rank 4/8/16 mask LoRA、A5 prototype LoRA、A6 uncertainty LoRA 与 A7 shuffled-prototype control。所有可训练 Adapter run MUST 使用共享 A/B 的低秩结构，A0--A7 对每个 epoch/sample 的 mask schedule MUST 一致；A7 的置换 MUST 只影响 Adapter condition 的 prototype state，且在 train、validation 中各自独立并与标签无关。

#### Scenario: A7 负对照保留非条件输入
- **WHEN** A7 在一个固定 sample_id 上评估
- **THEN** h_proto、mask 和 U0 base logits MUST 与 A6 相同，只有传入条件网络的 prototype state 来自固定 split 内置换映射

### Requirement: Full-data A0--A7 共用唯一 current U0
Full-pool capacity workflow MUST 从头训练一个且仅一个 current U0 checkpoint。该 U0 MUST 保留四模态 encoder、current fusion、64-Beam prototype/BPA、classifier 与 mixed-mask 训练，并禁用已退役 U1/U2/private/shared/CMSBL/U3/U4/U5/M1/M2/M3、teacher KL、two-stage 和动态 MoE 路线。A0--A7 MUST 全部加载同一 checkpoint SHA256，并保持各自已注册的 bias、rank、prototype condition 或 shuffled-control 定义。

#### Scenario: Stage 2 启动
- **WHEN** 唯一 U0 `last.pth` 已发布
- **THEN** A1--A7 MUST 冻结 U0 全部参数、保持 U0 eval、不更新 prototype，并共享 train、mask schedule、optimizer budget 与 validation
- **AND** checkpoint 缺失、哈希不一致或模型结构不符合 current U0 时 MUST fail closed

### Requirement: Full-data Adapter 训练预算和 Full 路径保持预注册
Full-data A1--A7 MUST 使用相同的最大 epoch、训练损失早停规则和 14 个非 Full mask 的确定性均衡 schedule，损失固定为交叉熵加 `1e-4 * ||delta_logits||_2^2`。实际 epoch 与 optimizer steps MUST 随结果记录。Full mask MUST 逐样本显式旁路，最大 logit 差不超过 `1e-7`、argmax mismatch 为零且 Top-1 差为零。

#### Scenario: 墙钟预算不足
- **WHEN** 动态测时预计约 8 小时内无法完成原拟 epoch
- **THEN** workflow MUST 同步减少 epoch
- **AND** 不得省略 15-mask 评估、Full 等价、资源审计或独立指标重算

#### Scenario: 用户取消严格墙钟上限
- **WHEN** 用户明确要求以收敛优先并取消严格 8 小时限制
- **THEN** Full-data U0 与 A1--A7 MUST 使用预先固定的 20 epoch 上限
- **AND** 用户随后因 GPU 稀缺明确要求早停时，U0 与 A1--A7 MUST 使用相同的 min-8、patience-3、0.5% 相对训练损失改善规则
- **AND** validation 仍不得用于早停、追加 epoch 或 checkpoint 选择
- **AND** 每项 MUST 保存实际 epoch、optimizer steps、最佳训练损失和 stop reason

### Requirement: Full-data U0 测时与正式训练隔离
Full-pool 的单 epoch 测时 MUST 使用不发布的 probe run。动态 epoch 确定后，正式 U0 MUST 从随机初始化重新开始，不得从 probe checkpoint 恢复。除数据划分与预注册的动态 epoch 数外，正式 U0 的模型和优化 profile MUST 与旧 3,600 对照使用的 clean U0 一致，包括关闭 router pattern bias、使用 BF16 且不启用 GradScaler、固定 seed、AdamW、weight decay 与 cosine scheduler。

#### Scenario: 正式 U0 发布前健康审计
- **WHEN** 正式 U0 `last.pth` 训练完成
- **THEN** workflow MUST 审计训练配置与 reference profile 完全匹配
- **AND** 64 个归一化 Beam prototype 的非对角两两余弦均值不得达到 `0.95`
- **AND** 任一条件失败时 MUST 在启动 A0--A7 前 fail closed

### Requirement: Full-pool 物理 GPU 绑定
Full-pool 编排器 MUST 以 `nvidia-smi` 报告的物理 GPU UUID 绑定任务，而不得假定 CUDA ordinal 与物理序号相同。当前 Stage 2 MUST 只允许用户提供的物理 GPU0/4/6/7；当新增 GPU 空闲时，恢复调度 MUST 只迁移尚未启动的任务，并跳过已完成或已运行任务，禁止覆盖或重复训练。

#### Scenario: 启动 GPU4 U0
- **WHEN** 编排器启动 GPU4 的 U0 probe 或正式训练
- **THEN** 子进程的 `CUDA_VISIBLE_DEVICES` MUST 是物理 GPU4 的 UUID，且 manifest MUST 同时记录物理序号和 UUID

### Requirement: Full-data ADBA-surrogate 对照保持单变量变化
完成 A0--A7 后，系统 MUST 支持 B1/B4/B6/B7 follow-up，分别保持 A1/A4/A6/A7 的结构、rank、条件输入和 shuffled-control 定义，只将 Adapter 分类目标改为 `0.5 * hard CE + 0.5 * circular soft-label CE`。soft label MUST 使用既有 64-Beam cyclic topology 与固定 `sigma=2.0`；损失 MUST 保留 `1e-4 * ||delta_logits||_2^2`。

#### Scenario: B6 prototype 语义判定
- **WHEN** B6 与 B7 完成同一 Full-pool validation 的 15-mask 评估
- **THEN** 两者 MUST 加载同一 U0 SHA256、使用同一 train、mask schedule、optimizer budget 与早停规则
- **AND** 系统 MUST 同时报告 ADBA、MAE、Within-3、Top-3、Top-1 与 Full 等价，不得只凭 B6 单项改善声称 prototype condition 有效

#### Scenario: ADBA-surrogate 不使用 validation 调参
- **WHEN** B1/B4/B6/B7 训练或停止
- **THEN** `lambda=0.5`、`sigma=2.0`、epoch 和 checkpoint MUST 不依赖 validation ADBA、Top-1 或其他 validation 指标

### Requirement: Mask-bias novelty triage 区分查表记忆与组合泛化
系统 MUST 支持冻结同一 U0 的 Global Bias、14-key Mask Lookup、现有 Mask MLP 和 Factorized Additive Bias。All-seen 比较 MUST 复用 B1，并使 Global/Lookup 使用相同 ADBA-surrogate、相同 schedule 前 8 epoch 和相同 optimizer steps。Full mask MUST 继续显式旁路。

#### Scenario: 条件启动 unseen-mask pilot
- **WHEN** 独立重算的 Mask MLP All-14 ADBA 严格高于 Mask Lookup
- **THEN** 系统 MUST 使用 seed 1、按 mask 基数分层的确定性 4-fold 划分启动且只启动 fold 0
- **AND** Mask MLP 与 Factorized Additive MUST 使用同一排除 held-out mask 的均衡 train schedule 和相同 8 epoch/steps
- **AND** 主泛化结论 MUST 只读取 held-out mask，并与 Frozen A0 对照
- **AND** 若 MLP 不高于 Lookup，unseen pilot MUST 标记为按预注册门槛跳过

#### Scenario: 未见 mask 不得进入拟合
- **WHEN** unseen fold 训练 Mask MLP 或 Factorized Additive
- **THEN** held-out mask MUST 在全部 train epoch 中出现零次
- **AND** validation 不得参与 fold、epoch、loss、checkpoint 或模型选择
