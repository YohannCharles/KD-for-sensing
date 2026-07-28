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

### Requirement: 环形 Beam 传输只表示局部概率算子
系统 MUST 支持 `circular_transport` 实验变体。它 MUST 对非 Full 行的 U0 softmax 概率应用每个缺失模态的局部环形转移核，而不得直接学习任意类别 logits 偏置。每个基础核 MUST 只覆盖相对 Beam 位移 `[-3, 3]`、非负且归一化；多个缺失模态 MUST 按 `[image, radar, gps, lidar]` 的固定顺序依次作环形卷积。Full 行 MUST 显式旁路，不得调用传输核。

#### Scenario: 传输保持概率质量和环形回绕
- **WHEN** 任意非 Full 行经一个或多个缺失模态核传输
- **THEN** 输出概率 MUST 非负且每行和为一
- **AND** 位移跨越 Beam 0/63 边界时 MUST 环形回绕
- **AND** 多模态结果 MUST 等于按固定模态顺序逐核传输的结果

#### Scenario: Full 传输严格恒等
- **WHEN** `circular_transport` 包装器接收 Full mask
- **THEN** 不得调用传输适配器
- **AND** logits MUST 与 U0 base logits 严格相同，最大绝对差不超过 `1e-7`

### Requirement: 环形传输与加性偏置使用同一 all-seen 比较协议
`circular_transport` 与 all-seen `factorized_bias` MUST 共享唯一 U0 SHA256、原始 14-mask schedule 的前 8 epoch、ADBA-surrogate、优化器步骤和固定 15-mask 评估。分析 MUST 独立重算 A0、B1、Factorized Bias 与 Circular Transport，并同时报告 All-14 ADBA、MAE、Top-1、Top-3、Within-3、Full 等价和每个基础核的概率质量。

#### Scenario: 比较完成但指标实现漂移
- **WHEN** 独立重算与任一 run 内部指标的最大绝对差超过 `1e-7`
- **THEN** 实验 MUST 失败，不得发布比较结论

### Requirement: Full-Pool BT-SCL 是独立的本地轻量模型工作流
系统 MUST 支持仅作为本地实验工具的 Full-Pool BT-SCL R0--R5 比较。六个方向 MUST 从同一 seed=2026 初始化训练相同的四模态轻量 CNN/MLP、固定时间-模态 token Fusion MLP 和 64-beam prototype bank；只有预注册损失和 R5 的固定 round-robin update phase 可不同。该模型不得使用 U0、Router、attention、Transformer、MoE、teacher/logit distillation、质量估计、动态损失权重、缺失重建或任意 logits residual。

#### Scenario: 所有方向共享结构与 nested views
- **WHEN** R0--R5 对任意基础样本训练或评估
- **THEN** 每个原始模态最多编码一次，并从同一 token tensor 构造严格 `S1 subset S2 subset S3 subset Full` 的四个 views
- **AND** Full view MUST 始终参与任务损失，all-missing MUST 显式失败

### Requirement: BT-SCL 拓扑损失必须绑定已审计的物理 codebook
R2--R5 MUST 绑定经 15-domain 回放验证的 `ula_dft_phase_cycle_v1` manifest、descriptor SHA256 与 64-beam phase order。它们 MUST 从该 order 生成环形 beam distance、连续 sector 和 local neighborhood；单独的 cyclic/circular Boolean 不得构成拓扑证据。

#### Scenario: topology provenance 无效
- **WHEN** topology manifest 缺失、哈希不一致、未覆盖 15 domain、没有 64-label 闭环或 label order 无法构造
- **THEN** R2--R5 MUST 在创建 optimizer 前失败

### Requirement: R6 使用标签锚定的拓扑随机占优
系统 MUST 支持不新增推理参数的 `r6_topological_stochastic_dominance` 本地筛选。R6 MUST 复用 R0--R5 的模型、初始化、Full-pool 数据、nested schedule 和 checkpoint selection；它 MUST 对 S1/S2/S3 使用真实 Beam 标签对应的 4/8/16-sector 监督，并对每个嵌套 transition 的半径 0/3/5 真实标签邻域概率施加 stop-gradient 随机占优约束。损失权重 MUST 固定为 hierarchy=0.25、dominance=1.0，不得由随机初始化 loss 反比例生成或根据 validation 调整。

#### Scenario: R6 机制与推理路径
- **WHEN** R6 对任意合法 nested chain 训练或评估
- **THEN** hierarchy 与 dominance MUST 只使用 Beam 标签和已审计 phase-cycle topology
- **AND** R6 MUST 不使用 Full-teacher KL、auxiliary head、Adapter、额外推理分支或动态权重
- **AND** 半径 0/3/5 的 violation 和 cumulative-mass delta MUST 与 R0 只读机制基线按相同口径报告

#### Scenario: 原训练 profile 持续失稳
- **WHEN** 已保留的 R6 轨迹显示 epoch-1 后 validation selection loss 持续上升且用户授权修改方案
- **THEN** post-hoc stable follow-up MUST 从同一初始化并行重跑 R0、R3、R6
- **AND** 三者 MUST 统一使用 encoder lr=`1e-5`、其余主模型 lr=`3e-5`、weight decay=`1e-3`、10% warmup、20 epoch 上限和 min-6/patience-4/0.1% validation early stopping
- **AND** R3 c2f/local 权重 MUST 固定为 0.25/0.2，不得继续使用随机初始化反比例权重
- **AND** 结果 MUST 标记为 post-hoc、single-seed、claim-ineligible

### Requirement: Candidate12 在共享 BPA 第一创新上比较 remix 与 motion
系统 MUST 支持本地 Candidate12 A0--A5 筛选。六路 MUST 从唯一 seed-2026、5-epoch warm-up checkpoint 开始，并共享 U0 encoder 配置、固定 MLP fusion、同一 `BeamPrototypeBank`、BPA temperature/权重/topology、Full-pool 数据、20 search epochs、optimizer steps 和 Full checkpoint selection。该 workflow MUST 不构建或调用 reliability router、动态模态权重、attention、MoE、logit residual、重建或禁用输入。

#### Scenario: BTPR-Mix 公平比较
- **WHEN** A1/A2/A3/A5 执行 remix step
- **THEN** 每个物理样本 MUST 只分配给一个模态且数据集不得扩张
- **AND** A2/A3/A5 MUST 读取 A2 生产的同一 15%--40% 容量约束 assignment SHA256
- **AND** A2 batch MAY 混合 assigned modalities，A3/A5 remix batch MUST 每批严格同模态，总 optimizer steps MUST 与 A0/A4 相同

#### Scenario: PAMR 非环形局部修正
- **WHEN** A4/A5 构造 motion 输出
- **THEN** signed order MUST 来自已审计 `principal_local_angle_deg` 的严格单调 64-label 顺序
- **AND** shift MUST 只覆盖 `[-3,3]`、不得端点回绕、每行概率和为一，超出范围的 residual 不得计算 offset CE
- **AND** final distribution MUST 是 anchor distribution 的 shift mixture，motion branch 不得直接输出 64 类 beam logits
- **AND** Candidate 2 的 overall gate MUST 同时要求 Dynamic Top-1 分别超过 Mean 与 Shuffle 至少 0.3 pp；任一失败时 MUST 停止 PAMR 候选

### Requirement: BTMA 因果消融隔离 assignment 因素
系统 MUST 支持 development-only 的 B0--B5 Full-pool BTMA 消融，并复用 Candidate12 的公共模型、BPA、seed-2026 warm-up checkpoint、20 search epochs、11,580 optimizer steps、1:1 joint/mixed assigned-modality batch、optimizer/scheduler 和 checkpoint selection。B0 不得读取标签或模型难度来决定分配；B1 MUST 从历史 A1 assignment statistics 读取精确全局模态比例，且不得计算逐样本 score；B2--B5 MUST 用各自当前模型的 train-only cache、模态内 percentile 和统一 15%--40% 容量分配器。B5 仅当在预注册容差内复现历史 A2 时，才可进入创新性判断。

#### Scenario: BTMA assignment 可审计且不泄漏 validation
- **WHEN** B0--B5 每两个 epoch 刷新 assignment
- **THEN** 系统 MUST 保存 sample_id、score、原始最佳 modality、容量修复和最终 modality
- **AND** 所有容量方法 MUST 对每个 modality 满足 15%--40%
- **AND** validation、outer test、channel、path、历史 beam 和未来 GPS MUST 不参与 assignment

### Requirement: BTMA 收尾只读且不得重开该路线
系统 MUST 支持一次只读 BTMA 收尾。它 MUST 从已保存的六个 BTMA checkpoint 按既有 `evaluate()` 口径重算 pattern=`full` 的逐样本 `anchor_logits`，MUST NOT 重新训练、改变协议或修改任何超参数。收尾 MUST 同时产出成对 temporal block bootstrap 与 assignment score 相关性，且报告 MUST 显式声明其结果不得用于重开 BTMA 作为第二创新候选。

#### Scenario: 报告 BTMA 方法间差异
- **WHEN** 收尾比较任意两个 BTMA 方法
- **THEN** 系统 MUST 同时给出点差与 `(domain, cav)` 内连续帧块的成对 bootstrap 区间
- **AND** 区间跨零时 MUST 表述为未超过对照，MUST NOT 表述为统计显著劣于对照
- **AND** 块长与重抽次数 MUST 在计算前固定，MUST NOT 依据结果调整

#### Scenario: 归因 assignment score
- **WHEN** 收尾计算 score 与单模态拓扑误差的相关性
- **THEN** 主表 MUST 只使用 epoch 5，即 score 与 warm-up train cache 严格同源的一轮
- **AND** 系统 MUST 另行报告跨 epoch 秩稳定性
- **AND** 由于 `capacity_repair_rate` 全程为 0，任何结论 MUST NOT 将收益归因于容量修复

### Requirement: Router 可观测性筛选只改变 routing 输入
系统 MUST 支持冻结 Full-pool U0 上的 Router 可观测性筛选。它 MUST 缓存与 mask 无关的 `latent_sequence` 及各 encoder 末层线性变换的输入特征，并在完全相同的冻结表征上只训练 router 与 quality 分支。encoder、projection、prototype bank、reliability head 与 classifier MUST 保持冻结且不产生梯度；U0 canonical recipe 与 public CLI MUST 不被修改。

#### Scenario: 启动缓存驱动的 router 训练
- **WHEN** 任一路线创建 optimizer
- **THEN** 从缓存重算的融合 logits MUST 与直接前向冻结 U0 在同一 mask 下逐样本一致
- **AND** 任一不一致 MUST 在创建 optimizer 前 fail closed
- **AND** 可训练参数 MUST 只包含 router 与 quality 分支

#### Scenario: 四条嵌套路线与负对照
- **WHEN** 系统构造 Q0--Q3 的 router 输入
- **THEN** Q0 MUST 只使用现有 router 标量与模态 one-hot，Q1 MUST 在其上追加每模态 prototype-space 状态
- **AND** Q2 MUST 在 Q1 之上追加投影前特征导出的 quality embedding
- **AND** Q3 MUST 与 Q2 参数量严格相等，且置换 MUST 只作用于投影前特征
- **AND** 结论 MUST 表述为 router 输入设计不足，MUST NOT 表述为 prototype 破坏了质量信息

#### Scenario: 预注册门槛判定
- **WHEN** 设定 N 与设定 C 的 Q0--Q3 全部完成
- **THEN** 方向成立 MUST 同时要求 Q2 优于 Q1、Q2 优于 Q3，且 MAE 与 within-3 不劣化
- **AND** 判据 MUST 使用三个 router seed 且区间不重叠，骨干单 seed MUST 在报告中写明
- **AND** 任一门槛不满足 MUST fail closed，MUST NOT 调参、追加 seed 或访问 outer test
- **AND** 处理组 MUST 报告冻结权重推理期消融的结果
### Requirement: 稀疏 pilot 只能驱动 prototype-space transition
系统 MUST 支持 development-only 的 Prototype-Conditioned Low-Overhead Pilot Transition。它 MUST 保留 U0 四模态 sensing forward、64 个一 beam 一 prototype 的 bank、15 个非空 sensing mask 与 audited cyclic topology；CSI 分支 MUST 只读取实际选择的 `M x Kp` 复 pilot observation，不得读取完整 CSI、path 参数、目标标签或未选择的候选 observation。

#### Scenario: CSI 可用的两阶段前向
- **WHEN** sensing prototype id 选择离线 lookup 中的 M 个 pattern
- **THEN** SparsePilotEncoder MUST 只接收这些 pattern 的 observation、id、frequency、valid mask 与 SNR
- **AND** final prediction MUST 是 `p0` 与局部/全局 prototype transition 的可靠性门控混合，不得是自由 beam-logit residual

#### Scenario: CSI 缺失或全部 pilot dropout
- **WHEN** `csi_available=false` 或没有有效 pilot token
- **THEN** `alpha` MUST 严格为 0 且 `p_final` MUST 与 sensing `p0` 逐元素相同

### Requirement: probe codebook 满足固定硬件约束
主 probe codebook MUST 使用确定性、split-independent 的恒模 Tx/Rx pattern pair；每个 Tx/Rx pattern 的范数 MUST 为 1，每个元素幅值 MUST 分别为 `1/sqrt(Nt)` 与 `1/sqrt(Nr)`。正式 validation/test MUST 只加载 train-time 已发布且 hash 匹配的 codebook 和 prototype lookup。

#### Scenario: 正式 lookup 推理
- **WHEN** 模型在 validation/test 或部署模式选择 pilot
- **THEN** 它 MUST 执行 prototype 到 pattern id 的 O(1) JSON lookup
- **AND** 不得运行 Gumbel、online search、oracle all-candidate selection 或基于测试 channel 重建 codebook

### Requirement: dense-to-sparse 训练只删除嵌套 pilot token
系统 MUST 支持 development-only 的 `32x16 -> 16x16 -> 8x8 -> 4x8` 连续预算课程。四阶段 MUST 共享冻结 U0、CSI encoder、PrototypeTransition、selector 参数和 optimizer 状态；Kp=8 MUST 是同一 16 点频率母网格的确定性子集，后续阶段不得更换 observation 定义。

#### Scenario: 从 dense 阶段转入目标预算
- **WHEN** 一个 curriculum 阶段完成并进入下一阶段
- **THEN** 新阶段交给 encoder 的 pattern 数和频点数 MUST 不增加
- **AND** D32x16 MUST 报告为 32 sounding/512 RE 的诊断上界，T4x8 MUST 报告为 4 sounding/32 RE 的目标结果
- **AND** validation 指标 MUST NOT 改变既定阶段、epoch、频点子集或 selector 训练状态

#### Scenario: matched-update 预算归因
- **WHEN** 系统比较 pilot 数量与 curriculum 效应
- **THEN** D32x16-only、4x16-only、T4x8-only 与 curriculum MUST 各训练总计 8 epoch 并共享初始化和训练 profile
- **AND** 4x16 MUST 只改变频率预算，T4x8 MUST 同时使用目标空间和频率预算
- **AND** 四路 MUST 分别绑定 GPU0--3，失败不得触发抢占、自动调参或隐式重跑

#### Scenario: 增加样本与更新量后的收敛归因
- **WHEN** 用户授权检验短程训练是否不足
- **THEN** 四路 MUST 各使用 2,000 个 domain-balanced train、1,000 个 domain-balanced inner validation、40 epoch 与 batch 8
- **AND** 单阶段 arm MUST 获得 40 epoch，curriculum MUST 按四阶段各 10 epoch 且继承同一 optimizer
- **AND** 系统 MUST 逐 epoch 记录分项 loss、训练决策变化、route/reliability 分布和模块梯度范数
- **AND** validation MUST 分块只读且不得决定 epoch、checkpoint 或超参数

#### Scenario: 严重模态缺失下由 sparse CSI 兜底
- **WHEN** 用户要求 sparse CSI 在只剩 1--2 个感知模态时保持 beam prediction 性能
- **THEN** train mask MUST 按可用模态数使用固定 `50%/35%/10%/5%` 权重，并在同一 cardinality 内均衡各 mask
- **AND** 模型 MUST 产生不读取 sensing feature 的 CSI-only prototype distribution，并对其施加直接 train-only 分类/拓扑监督
- **AND** reliability gate MUST 显式读取 sensing availability fraction，且 CSI unavailable 或全部 pilot dropout 时仍严格输出 `p_final=p0`
- **AND** 主判据 MUST 是 Single Macro、Single Worst 与 All-14 Macro 的 CSI-on 相对 CSI-off Top-1 增益，并同时要求 Full Top-1 不退化

#### Scenario: 逐级稀疏定位兜底预算断点
- **WHEN** D32x16 有弱严重缺失增益而 4x16/4x8 没有增益
- **THEN** 系统 MUST 以独立 40-epoch arm 补齐 D16x16、S8x16、S16x8 与 S8x8
- **AND** 四路 MUST 保持模型、loss、2,000/1,000 样本索引、missing-mask schedule、seed 与 batch 8 不变
- **AND** S8x16 与 S16x8 MUST 同为 128 RE，以区分空间 sounding 与频率 token 稀疏的影响
- **AND** 结果 MUST 与既有 D32x16、4x16 和 4x8 一起按 512/256/128/64/32 RE 报告，不得用 validation 选择或重训预算

### Requirement: CSI 信息诊断必须隔离复杂决策模块
恢复诊断 MUST 使用跨 split/seed 相同的固定 4-pattern QPSK codebook 和 8 个均匀频点。CSI-only 任务 MUST 仅包含 `SparsePilotEncoder`、可选两层时序 GRU 和 64 类 MLP；concat 任务 MUST 仅追加冻结 U0 `z_sensing`，不得启用 selector、local/global route、reliability gate 或 preserve loss。

#### Scenario: 比较当前与历史 CSI 信息
- **WHEN** 系统运行 I1--I5 信息诊断
- **THEN** I1 MUST 预测当前 `beam_t`，I2 MUST 由 `CSI_t` 预测 `beam_{t+1}`，I3 MUST 由 `CSI_{t-4:t}` 预测 `beam_{t+1}`
- **AND** I0/I4/I5 MUST 在同一物理 validation 样本和相同 single masks 上比较冻结 U0、单帧 concat 与历史 concat
- **AND** I1 的 `beam_t` MUST 直接来自既有 `beam5` 功率 artifact 的 argmax；Full-pool `beam_label` 是未来标签别名，MUST NOT 用作当前标签

#### Scenario: 决定是否进入 forced transition
- **WHEN** 三 seed 信息诊断完成
- **THEN** 若 I1 很弱，workflow MUST 停止并优先审计 simulator/标签/频率/码本
- **AND** 若 I4/I5 均未方向一致地超过 I0 Single Macro，workflow MUST NOT 启动 Stage A1/A2、route 或 selector
- **AND** 若 I5 优于 I4，后续 forced transition MUST 使用五帧历史 CSI；否则使用单帧 CSI

### Requirement: trajectory recovery 按三轮单因素闭环推进

Trajectory recovery MUST 先以 32x16 固定 observation 运行 I1--I5；若 I3 明显高于 I4/I5 的 Single Worst，MUST 从 dense I3 checkpoint 初始化并逐步比较 16x16、8x16、16x8、8x8、4x16、4x8，其中两个 128 RE 与两个 64 RE 配对 MUST 分别报告。稀疏任务 MUST 重置 optimizer/scheduler。最后 MUST 在选定预算上评估确定性 availability fallback：可用感知模态数不超过 2 时使用 CSI-only 分布，可用 3 个模态或 Full 时使用冻结 M4。每一轮 MUST 在下一轮启动前生成分析；Round 3 MUST 同时报告 Full、Single Macro/Worst 与 All-14 Macro/Worst，且不得引入可学习 gate、route 或 selector。

#### Scenario: 选择最小可行 pilot budget

- **WHEN** Round 2 完成多个嵌套预算
- **THEN** workflow MUST 以 CSI-only I3 Top-1（集成后四个 single-mask 的共同 Macro/Worst）为第一目标，并在距最佳值不超过 0.5 个百分点的候选中选择最小 RE
- **AND** Full Top-1 MUST 作为灾难性退化 guard 单独报告，但 MUST NOT 参与严重缺失主排序
- **AND** 若没有预算满足，MUST 明确报告失败并只把最佳预算当作诊断上界
- **AND** 同一 validation 上的三轮自适应结果 MUST 标记为 development exploratory evidence

#### Scenario: 严重缺失使用确定性 CSI 旁路

- **WHEN** Round 3 集成所选 sparse I3 checkpoint
- **THEN** 可用感知模态数不超过 2 的样本 MUST 使用 CSI-only 分布
- **AND** 可用 3 个感知模态或 Full 的样本 MUST 原样使用冻结 M4 分布
- **AND** Full 概率最大绝对差和 argmax mismatch MUST 均为零

### Requirement: 恢复训练必须保留完整收敛与恢复状态
恢复诊断及后续 A1/A2 MUST 逐 epoch 保存训练损失分项、validation Top-k/Within-3/MAE、Single/All-14、Fix/Harm、适用的 alpha/route 分布、模块梯度、KL/entropy 与学习率。最大 epoch MUST 为 60，统一 early-stopping patience MUST 为 10。

#### Scenario: 保存恢复 checkpoint
- **WHEN** 任一学习任务完成一个 epoch
- **THEN** last checkpoint MUST 包含 model、optimizer、scheduler、epoch、resolved config 与 Python/NumPy/Torch/CUDA RNG state
- **AND** workflow MUST 由 Single Worst 驱动 early stopping 并发布 best Single Worst，同时分别维护 best Single Macro、best Fix Rate 和 best validation loss checkpoint；不适用的选择指标 MUST 在 metadata 中明确标记
- **AND** All-14 与 Full MAY 只在最终 best Single Worst checkpoint 上补算，不得取代严重缺失主指标
