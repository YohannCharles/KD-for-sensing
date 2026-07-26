## Context

当前 Clean MMW U0 是四模态 masked-mean supervised router，使用 `BeamPrototypeBank` 产生 64 维 Beam logits。U0 已有 prototype 分支，但其 forward 只返回融合特征和 logits，未导出可供外部条件化的 prototype 关系。实验必须加载已审计的 Seed 1 U0 checkpoint，只使用 3600 条 inner_train 优化 Adapter，并在相同 900 条 inner_validation 上报告 15 个非空掩码；outer test 不可访问。

首轮 3,600/900 筛选完成后，需要判断排序是否受样本规模限制。Town3 的 15 个 domain 共提供 46,860 个候选窗口。候选 CSV 的 `contiguous_segment_id` 按 CAV 定义，但 Radar 与 BS-GPS 是同一 domain 内跨 CAV 共享的 RSU 时间资源，因此按 CAV trajectory 拆分会产生原始资源泄漏，且每个 domain 只有 3--4 个长 CAV 段，无法可靠实现覆盖所有 domain 的 80/20 group split。Full-pool 验证必须改用每个 domain 全部 CAV 对齐的共享连续时间轴。

## Goals / Non-Goals

**Goals:**

- 以不改变 U0 参数、状态或 Full logits 的方式导出只读 prototype state。
- 为非 Full mask 增加零初始化、共享投影矩阵的轻量低秩决策残差，并完成 A0--A7 的确定性对照。
- 使训练、评估、逐样本导出、独立重算与配对统计可由本地协议和 checkpoint fail-closed 复现。
- 在不访问 outer evidence 的前提下构建最大利用率 Full-pool 开发协议，并以预注册训练损失早停完成唯一 Full-data U0 及 A0--A7 对比。

**Non-Goals:**

- 不重新训练 U0 编码器、projection、融合、prototype 或 classifier；不改变 canonical MMW recipe 或公共 CLI。
- 不实现蒸馏、模态生成、动态加权、MoE、新原型聚类、Full 性能优化或自动 Stage B 执行。
- 不读取 outer test，或将 validation 用于早停、超参数选择、可拟合归一化状态或 checkpoint 选择。
- 不构造 Full-pool 学习曲线，不为各 Adapter 分别训练 U0，也不因 Full-data validation 结果调整 rank、条件输入、损失或学习率。

## Decisions

### 只读 prototype state 复用现有 prototype logits

`BeamPrototypeBank` 以归一化融合特征与已训练 prototype 的余弦相似度产生 logits。新增 `describe(features)` 在同一计算中返回 logits、soft assignment、nearest id、nearest distance、最近/次近距离差、熵和输入到最近 prototype 的残差范数。U0 在调用点将 `output_features` 送入此接口，返回值仅写入输出诊断字段。

选择此方案而非建立新的聚类或学习状态编码器，因为它对 U0 的预测数值没有影响、无标签输入且可在 `no_grad()` 下复用同一组 prototype。Adapter 在进入条件网络前再次 `detach()` 全部连续 state，确保没有通往 U0 的梯度路径。

### 采用包装器而不是修改 U0 的分类路径

`FrozenU0DecisionAdapter` 持有已加载的 U0 和可训练 Adapter。包装器让 U0 永远 `eval()`，在 `torch.no_grad()` 中计算 base 输出，显式判断每一行 mask；Full 行直接选择 base logits，非 Full 行才相加 `delta_logits`。这样可对混合 batch 保持逐样本 Full 等价，而非依赖可学习 gate 的近似关闭。

选择包装器而不是把 Adapter 注入 U0，因为后者会扩大 U0 主线 forward contract，并增加 checkpoint/优化器意外包含 U0 参数的风险。

### 共享低秩方向，以小条件投影表达实验差异

基础 Adapter 用共享 `A: d -> r` 和 `B: r -> 64`，计算 `B(alpha * A(h_proto))`。A1 仅输出 mask bias；A2--A4 的 `alpha` 只由 4 位 mask 得到；A5 使用 mask 与压缩的 prototype assignment；A6 再加入四项不确定性；A7 仅将 condition branch 的 proto state 按 split 内、sample_id 确定性置换。A 使用小随机初始化，B 全零初始化。

选择共享 A/B 而不是每个 mask 一套 head，可直接检验“条件控制低秩方向”而非通过参数量扩大拟合能力。A7 不改变 h_proto、mask 或 base logits，提供样本级原型语义的负对照。

### 单一受协议约束的专用实验工作流

新工作流使用现有 clean protocol loader 与 U0 config 构建函数，在创建模型/optimizer 前重新运行 fail-closed 审计。它预先写入基于 sample_id 的 20-epoch mask schedule，使 A0--A7 对每个 epoch/sample 读取同一非 Full mask；归一化连续 prototype 条件的统计只由 inner_train 拟合。验证一次性遍历 15 个固定 mask，输出高精度 NPZ，随后由独立 NumPy 工具重算指标、bootstrap 和 McNemar。

选择专用 workflow 而不是修改泛化训练器，是为了避免把实验性 Adapter 误接入 public CLI、canonical U0 优化器或其 validation/checkpoint 策略。

### Full-pool 采用 domain-wise contiguous macro split

每个 domain 先按真实 `window_start_frame`、历史帧集合与 target frame 建立共享时间轴，并在约 80% 的时间位置设置唯一边界。所有依赖资源完全位于边界前的窗口进入 train，完全位于边界后的窗口进入 inner validation，跨界窗口才被 purge。划分后按 sample、target、完整行、窗口与目标帧、camera、lidar、radar、UE GPS、BS GPS、channel/beam 资源重新执行零交叉审计。

588 个历史验证窗口不从已锁定 outer-evidence CSV 恢复。现存 `mmw_twc_outer_v1` manifest 已将相关 group 标记为 `excluded_development`；协议构造器将这些公开 block 范围映射回其 source strict-train CSV，并要求每域计数与 manifest 的 `excluded_window_count` 一致、总数严格等于 588。恢复出的身份只禁止进入新 train；若自然落入新 validation，可以作为只读开发样本保留。

选择 macro split 而非 trajectory split，是因为同一 RSU frame 的 Radar/BS-GPS 会跨 CAV trajectory 共享。选择单一边界而非旧 7 帧 block，是为了只损失真实边界上的跨界窗口，避免周期性 guard 导致的数据利用率下降。

### Full-data 采用唯一 U0 的两阶段流水线

GPU4 从头训练一个 current U0，训练 epoch 上限由正式运行前的吞吐、单 epoch、单 mask 评估、checkpoint 加载和 Adapter 短段测时共同确定。测时和正式运行都不读取 outer test。U0 只发布一个带 SHA256 的 `last.pth`；Stage 2 必须先验证该哈希，然后在用户提供的物理 GPU0/4/6 启动 A0--A7。GPU0 顺序执行 A0/A1/A3，GPU4 顺序执行 A2/A4，GPU6 先执行 A5；当用户确认 GPU7 空闲后，尚未启动的 A7 迁移到 GPU7，当 A5 完成释放 GPU6 后，尚未启动的 A6 从 GPU4 队列迁移到 GPU6。恢复调度器只跳过已有完整 `metrics.json` 的任务，或接管旧 manifest 及 `/proc` 中唯一匹配且状态仍为 running 的任务，禁止覆盖或重复启动。

A1--A7 使用同一个固定 20-epoch mask schedule、相同训练窗口、loss、optimizer 和预注册早停规则；实际只消费截至各自停止 epoch 的 schedule 前缀。U0 始终冻结并保持 eval，prototype 不更新。validation 只在训练结束后固定枚举 15 个非空 mask，不参与早停或 checkpoint 选择。U0 与 A1--A7 的最大 epoch 均为 20，最少训练 8 epoch；监控各自 epoch 聚合训练主损失，相对最佳值改善不足 0.5% 计为未改善，连续 3 epoch 未改善后停止。每项必须报告实际 epoch、optimizer steps、最佳训练损失与 stop reason。该规则在重启前固定，不依据已见 validation 结果或后续曲线修改。

### ADBA-surrogate 只改变 Adapter 训练目标

完成 A0--A7 后追加 B1/B4/B6/B7，不重训 U0、不覆盖 A 组，也不把 ADBA 用作 validation 早停或 checkpoint 选择。B1/B4/B6/B7 分别复用 A1/A4/A6/A7 的结构、rank、条件输入和 shuffled control，并对同一 logits 使用 `0.5 * CE(hard) + 0.5 * CE(circular_soft_label)`；soft label 复用现有 64-Beam cyclic topology，固定 `sigma=2.0`。损失继续包含 `1e-4 * ||delta_logits||_2^2`，其余 AdamW、cosine、mask schedule、min-8/patience-3/0.5% 训练损失早停与 15-mask 评估保持不变。

硬 ADBA 包含离散 top-k，不能作为直接反向传播目标；circular soft-label CE 是预注册 surrogate。主分析以 All-14 ADBA、MAE、Within-3 和 Top-3 为主，Top-1 作为约束，同时必须比较 B6 与 B7，避免将普通 logits 平滑误判为 prototype 条件收益。B1/B4/B6/B7 分别绑定物理 GPU0/4/6/7，全部产物写入独立 `outputs/full_pool_capacity/adba_surrogate/`。

本轮 Full-pool 编排使用 `nvidia-smi --id=<physical index>` 解析 UUID，再把 UUID 写入 `CUDA_VISIBLE_DEVICES`；同时固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。这是因为裸 CUDA ordinal 在当前机器上不等于 `nvidia-smi` 的物理序号。当前 Stage 2 只允许用户最新提供的物理 GPU0/4/6/7。

### Mask-bias novelty triage 先比较记忆能力，再测试组合泛化

后续单 seed 筛选不重训 U0，也不覆盖 B1。All-seen 阶段复用 B1 的 Mask MLP checkpoint，并新增 Global Bias 与 14-key Mask Lookup；三者使用相同 ADBA-surrogate、Full-pool train、原 mask schedule 的前 8 epoch 和相同 optimizer steps。8 epoch 来自 B1 已预注册、只读取训练损失的停止点，固定后不再对新方法使用 validation 选择训练长度。

只有独立重算的 MLP All-14 ADBA 严格高于 Lookup 时，才启动 unseen pilot。14 个非 Full mask 先按可用模态数 1/2/3 分层，在每层使用 seed 1 确定性置换并轮转分配到 4 fold；本轮只运行 fold 0，不按已见 validation 结果手选 mask。训练 schedule 在保留全部 train sample 和 optimizer steps 的前提下，仅从非 held-out mask 中重新均衡生成。验证仍固定枚举全部 15 mask，但主结论只读取预注册 held-out mask。

Lookup 对未见 key 没有可学习输出，因此 unseen pilot 使用 Mask MLP 与 `b(m)=b0+sum_i (1-m_i)e_i` 的 Factorized Additive Bias 对照，并同时报告 Frozen A0。先做一个 fold、一个 seed；多 seed 和其余 fold 本轮不自动执行。当前 B1 权重另做无标签 weight-space compositionality probe，该 probe 不替代 held-out retraining 或预测指标。

### 环形 Beam 分布传输避免任意 logits 残差

Mask MLP 相对 Lookup 的收益不足以单独证明任意 logits 残差具有独立方法创新，且 Factorized Additive Bias 已解释其中大部分效果。因此新增 `circular_transport` 只在非 Full mask 上作用于 `softmax(base_logits)` 的概率分布，而不直接预测自由 logits 偏置。每个缺失模态 `i` 学习一个定义在环形相对位移 `[-3, 3]` 上的非负、归一化局部核 `K_i`；模态顺序固定为 `[image, radar, gps, lidar]`，多个缺失模态依次作圆周卷积。每一步保持总概率为一，因而组合核表达局部偏移和不确定性扩散，而不是任意类别方向。

Full mask 必须由包装器显式旁路，既不调用传输适配器也不计算 kernel。非 Full 输出的训练 logits 为传输后概率的 log，仍使用已经固定的 ADBA-surrogate 和 `1e-4 * ||delta_logits||_2^2`，不新增基于 validation 的超参数选择。每个基础核以固定的中心偏置初始化，使初始算子接近恒等但保留对所有局部位移的梯度。

本轮只进行 all-seen、单 seed 的可行性检验：`circular_transport` 与新训练的 `factorized_bias` 均使用原始 14-mask schedule 的前 8 epoch、同一 U0 SHA256、同一优化器步数、同一 ADBA-surrogate 和 15-mask 评估。实验产物不覆盖既有 mask-bias triage；独立分析必须重算 A0、B1、Factorized Bias 与 Circular Transport 的逐 mask 指标，报告概率守恒/Full 等价，并将 MAE 与 ADBA 同时作为判据。

Full-pool pooled dataset 的 15 个 domain 构造使用标准库线程池并行。每个 domain 的 prepared CSV 先向量化检查必填资源单元、Radar `_RA` 约束与 Beam 标签，再按真实资源路径去重；只有去重后的 camera/lidar/radar/UE GPS/BS GPS（含 Radar `_DA`）路径进入并行 `is_file` 校验。15 个 domain 各最多使用 6 个资源校验 worker，使 pooled 构造的总资源校验并发不超过 90；任何缺失、非法或越界单元仍按首个确定性位置 fail closed。train-only GPS scaler 将每个 train leaf 的 float64 sum/sum-of-squares 独立并行计算，再严格按 manifest domain 顺序归并。这样只改变吞吐，不改变 train-only 边界、样本集合、确定性顺序或 validation 只读契约。

Full-pool split audit 不再对每行重复扫描全部列。sample/target/trajectory 与各资源族使用逐列向量化集合构造，历史/目标 frame JSON 和完整行 SHA256 使用 `itertuples` 保持逐样本覆盖；输出仍是相同的强哈希身份集合及 train-validation 零交叉判据。

### Full-pool 复用确定性帧缓存并将 LMDB 置于测量门槛之后

现有 `outputs/cache/MMW/<condition>/image_derived` 与 `lidar_bev` 已由 15 个 `all_sequences.csv` 的原始资源生成。正式工作流只在以下条件全部满足时以只读方式复用：新 train/validation 引用资源覆盖率为 100%、缓存参数与当前 ImageNet RGB/LiDAR BEV 变换一致、代表性逐元素重算误差为 0、缓存复用 manifest 绑定 source CSV、Full-pool protocol fingerprint 与当前变换代码 SHA256。正式 DataLoader 对 cache miss、metadata mismatch 或 shape mismatch fail closed，不得静默回退到原始转换。

GPS 原始坐标属于无标签确定性输入，可按 domain 预先保存为紧凑 NPZ 并由 DataLoader worker 只读加载。GPS scaler 仍只由 Full-pool train 拟合；测时 probe 产生的 scaler 只有在其 metadata、sample identity hash、sample count、artifact SHA256 与同一 Full-pool protocol 完全一致时才能被正式 U0 复用。validation 坐标可被缓存，但不得进入 scaler moments。

旧 split-level LMDB 绑定历史 split、缺少 Radar 且按重叠窗口重复完整五帧样本，因此不得用于本轮。先测量严格 NPY cache DataLoader 吞吐；只有其稳定吞吐低于已测 U0 训练消费速率的 1.5 倍时，才允许构建按 15 个 domain 分片、只存唯一帧且不包含标签或归一化状态的新 LMDB。该门槛通过时必须跳过 LMDB，避免无证据复制约 54GB 派生张量。

测时 epoch 只用于估计吞吐，写入独立 `u0_timing_probe`，不得作为正式 U0 的 epoch 1。动态预算确定后，`u0_seed1` 从相同 seed 的随机初始化重新训练，以免 1-epoch probe 的 optimizer、AMP 或 scheduler 终态污染正式轨迹。Full-data 派生配置显式固定旧 3,600 clean U0 的对照 profile：关闭 router pattern bias、BF16 无 GradScaler、AdamW、weight decay 3e-4、cosine warm restart、temporal-missing seed 1；不从本地旧 checkpoint 或 outputs 读取配置。

正式 checkpoint 发布前计算归一化 prototype 的非对角两两余弦统计。旧 clean U0 的均值约为 -0.01，而失败 run 为 0.99956，因此以 0.95 作为仅检测明显塌缩的 fail-closed 阈值。该检查不读取 validation，也不用于选择 checkpoint 或调参。

### Full-Pool BT-SCL 使用专用轻量模型与共享 nested chain

BT-SCL 是独立于冻结 U0/Adapter 的 single-seed development workflow。它从 Full-pool train 的同一公共初始化出发，使用 Image、LiDAR、Radar 和 GPS 的专用轻量 encoder；每个 encoder 一次性产生 `[B,T,D]` pre-prototype tokens，随后仅在 Fusion 前复制为 `S1,S2,S3,Full` 四个 mask view。Fusion 是固定 token 顺序的 LayerNorm/MLP，不使用 Transformer、attention、Router、quality/reliability estimator、特征重建或 logits residual。每个 view 共享一个 L2-normalized 64-beam prototype bank，prototype 只由主融合任务更新。

训练 schedule 在 train split 上预生成且平衡四种起点、六种双模态和四种三模态集合；R0--R5 共用 schedule、seed=2026、初始化、数据、优化器、30 epoch 预算和 checkpoint selection。R1 的单模态 auxiliary head 仅可将对应 token 映射到 detached prototype bank，不能读取 Fusion 或其他模态。R2 使用多模态风险相对少模态风险的 stop-gradient hinge；R3 使用由 phase-cycle sector 聚合的 coarse-to-fine KL；R4 组合二者；R5 在固定 `I:L:R:G:Joint=1:1:1:1:4` 的 round-robin 参数更新日程中组合三项预注册损失。每个 epoch 的 validation 只向量化累计统一 checkpoint selection 所需的 15-pattern CE；完整 Top-k、MAE、分域、天气、sector、证据、单调性和粗细一致性诊断仅在加载最佳 checkpoint 后只读计算一次。validation 不参与 loss scale、schedule、学习率或早停决策。

### R6 使用标签锚定的层次监督与多半径随机占优

R0--R5 的 single-seed 结果显示 R3 提高 All-14 和 Double Top-1，但同时显著损害 Within-3、MAE 与 distance>5；其随机初始化校准还因近均匀分布下 KL/monotonicity 数值很小而生成过大的固定权重。R6 因此不继续放大 teacher consistency，而将拓扑目标直接绑定 ground-truth Beam。

R6 对 S1、S2、S3 分别计算真实 Beam 所属 4、8、16 sector 的负对数概率，并取宏平均作为 `L_hierarchy`。对每个 `S subset T` 和固定半径 `r in {0,3,5}`，计算真实 Beam 拓扑邻域累计概率 `q_r`，使用 `relu(stopgrad(q_r(S)) - q_r(T))` 得到 `L_dominance`。总损失固定为 `L_base + 0.25 * L_hierarchy + 1.0 * L_dominance`；这些有界权重在实现和正式运行前注册，不读取 validation，也不使用随机初始化 loss 的反比例缩放。

R6 复用 R0--R5 的模型、公共初始化、37,038/9,180 Full-pool 协议、nested schedule、AdamW 参数组、30 epoch、统一 validation CE checkpoint selection 与完整 15-pattern 评估。它不增加 auxiliary head 或推理参数。除原有指标外，R6 必须报告半径 0/3/5 的逐 transition violation rate 和 cumulative-mass delta；R0 checkpoint 只读补算同口径机制基线。R6 仍是 single-seed development evidence，只有通过原 5/6 gate 且距离指标同步改善时才可建议 multi-seed。

### 失稳后 follow-up 使用统一稳定训练 profile

首条 R6 正式轨迹沿用 R0--R5 profile，并在 epoch 1 后持续出现 train loss 下降而 validation selection loss 上升；用户据此明确授权停止并修改方案。该轨迹在 epoch 19 后停止并完整保留为 `interrupted_unstable`，不得作为完成结果或被覆盖。

稳定 follow-up 从同一公共随机初始化分别训练 R0、R3 和 R6，保持模型、Full-pool 协议、nested schedule、batch size、checkpoint selection 和全部评估一致。三个分支统一使用 encoder lr=`1e-5`、projection/fusion/prototype lr=`3e-5`、AdamW weight decay=`1e-3`、10% warmup、cosine decay、20 epoch 上限；R3 不再使用随机初始化反比例权重，固定 c2f=`0.25`、local=`0.2`。从 epoch 6 起，validation selection loss 相对历史 early-stop best 改善不足 0.1% 连续 4 epoch 时统一停止，最终仍加载绝对最低 selection loss checkpoint。

该 stable profile 是看到开发验证失稳后的 post-hoc 筛选，允许使用 inner validation early stopping，但 outer test 继续锁定，结果保持 single-seed、claim-ineligible。只有 R6 在相同 stable R0/R3 下通过原 5/6 gate，才可另行预注册无 post-hoc 修改的 multi-seed 复验。

### 物理 phase-cycle topology 是 BT-SCL 的强制协议输入

R2--R5 启动前必须读取并验证本地 `ula_dft_phase_cycle_v1` topology manifest、descriptor SHA256、15-domain metadata consistency、64-label bijection 和端点闭环边。通过 audit table 的 phase order 建立 `D[i,j]=min(|pos_i-pos_j|, 64-|pos_i-pos_j|)`；R2 使用 `D/32`，R3 用相同顺序构造 4/8/16 contiguous sectors 和 `D<=3` 的 local neighborhood。这只表达本地 ULA-DFT codebook phase 邻接，不主张世界方位角。manifest 不存在、哈希不符或 audit 不完整时，R2--R5 必须 fail closed；R0/R1 可报告 topology unavailable 而不计算这些损失。

### Candidate12 复用 BPA 但排除 current U0 router

Current U0 的 `BeamPrototypeBank` 与 BPA 是可信第一创新实现，但其 Full 融合使用 sample-specific supervised reliability router，违反本轮禁止动态模态权重、reliability router 和 quality estimator 的约束。因此 Candidate12 作为独立本地 workflow，复用 U0 的四模态 encoder 配置、共享 `BeamPrototypeBank`、temperature=`0.1`、BPA `lambda_proto=0.2`、`lambda_modality_proto=0.1`、soft-label sigma=`2.0` 与审计 `ula_dft_phase_cycle_v1`；融合改为固定 token 顺序的 LayerNorm/MLP，所有 A0--A5 完全一致。它不加载 U0 checkpoint、router 参数、历史 beam、channel、path 或未来 GPS，也不成为 canonical U0。

唯一 warm-up 使用 seed 2026、Full-pool 37,038 train、完整四模态输入与公共 BPA 损失训练 5 epoch，并发布一个 checkpoint。A0--A5 从同一 SHA256 继续相同的 20 search epochs、AdamW/cosine/BF16、batch size、总 optimizer steps 和 `CE_full + 0.25 * topology_risk_full` checkpoint selection；不早停，不用 validation 改 assignment、K、损失或 optimizer。

KL assignment 由各 A1 checkpoint 每两 epoch在 train-only eval cache 上确定性更新。Prototype-risk assignment 必须对每模态的 topology risk 与 true-prototype margin hardness 分别做 train-only percentile rank，并满足每模态 15%--40% 容量。为保持 A2/A3/A5 assignment 完全一致，A2 是唯一共享 risk-assignment 生产者：warm-up 先写 epoch 005，A2 每两个 search epoch写 epoch 007/009/...；A3/A5 在进入下一阶段前只读等待同一哈希文件。A2 使用 mixed assigned-modality batch，A3/A5 使用按 modality 且在 domain/weather/8-sector/head-tail 内确定性轮转的 homogeneous batch。

PAMR 的 signed order 不使用闭环 phase order，而从同一 topology table 的 `principal_local_angle_deg` 由小到大构建 64 标签唯一、严格单调的非环形顺序。`K=3` shift 越界质量置零后归一化，禁止端点 wrap。Motion branch 只能读取历史 projected sensor tokens并混合 anchor 分布；远于 K 的训练样本不计算 offset CE。A4/A5 还必须执行 Dynamic、Zero、train-mean、validation-shuffled 与 Oracle-local 五种只读推理替换。

Candidate 2 除通过计数门槛外，还必须同时满足 Dynamic 相对 Mean 和 Shuffle 的预注册 `+0.3 pp` 门槛。若任一失败，按协议的“Dynamic 约等于 Mean/Shuffle”解释 fail closed，禁止仅凭 Full、Oracle 或天气项凑足计数后推荐 PAMR。

### BTMA 必须先完成因果消融才可作为第二创新候选

Full-pool BTMA 不改变 Candidate12 的模型、BPA 损失、warm-up、优化器、20 epoch/11,580 step 日程或 checkpoint selection，只替换 mixed assigned-modality step 的每样本 modality assignment。B0 为固定 sample-id hash 的随机均衡 assignment；B1 只复用历史 A1 实际全局比例；B2--B5 分别为 KL、topology risk、prototype margin 和 0.5/0.5 risk+margin，并在每两个 epoch用当前分支的 train-only 单模态 cache 更新。除 B1 外均使用同一确定性 15%--40% 容量修复器；B1 不得计算逐样本难度。B5 必须在预注册容差内复现历史 A2，否则所有创新性判断 fail closed。

最终报告区分“实用方法推荐”和“可主张的新第二创新”：Literature-style KL Data Remixing 可以作为前者，但不能在 BTPR-Mix、PAMR 与组合均失败时被重新命名为本项目的新创新。KL assignment 的 Radar/GPS collapse 按两者合计比例判断，同时保留单模态是否超过 80% 的独立口径。

### BTMA 收尾只做只读封档，且不得在负方向过度断言

BTMA 未保存逐样本 validation logits 的原因不是架构限制：`_validation_predictions()` 已实现但从未被调用。六个
`best_checkpoint.pt` 与确定性 validation loader 都在盘上，因此收尾是一次只读重推理，不重训、不改协议。重算必须复用
`evaluate()` 的口径：B0--B5 均非 motion 方法，主指标取 pattern=`full` 的 `anchor_logits`。

配对区间以 `(domain, cav)` 内长度 32 的连续帧块为重抽单元，块长与重抽次数在计算前固定。这一步的目的是防止把
“B5 没有赢” 误写成 “B5 输了”：B5--B0 的 Top-1 与 Within-3 点差分别为 -0.59 pp 和 -0.76 pp，而 9,180 样本上的朴素
1σ 约为 0.44 pp 与 0.52 pp，且验证窗口是连续帧、有效样本量更低。预期结论是两者不可区分，据此对 BTMA 的否决理由
必须落在“打不过零成本对照”与“机制自证失败”，而不是统计显著的劣势。

score correlation 是纯后处理，不运行模型。epoch 5 的 assignment score 正是由已发布的 warm-up train cache 算出，
是唯一严格对齐的一轮，因此以该轮的每模态 Spearman(score, 单模态环形拓扑误差) 为主表；后续 epoch 的 score 对同一
warm-up 误差只作为追踪稳定性的参考，不得当作等价证据。另外必须报告 Spearman(score@t, score@t+2)，用以量化
`assignment_statistics.csv` 中 B3/B5 高达 0.86--0.99 的 `change_rate` —— 该值高于四模态独立随机重抽的 0.75 期望，
说明 score 是反持久振荡而非稳定课程。`capacity_repair_rate` 除 b4 两个 epoch 外全程为 0，因此任何结论都不得把收益
归因于容量修复。

### Router 可观测性通过冻结表征缓存实现严格因果隔离

诊断的六个假设中只有 Router observability failure 成立，可辩护的主张是 **router 输入设计不足**，不是
**prototype 破坏了质量信息** —— 后者已被检验且在 C7 聚合层面仅 0.012 drop、报告明确标注不支持。所有产出文档按
前者措辞。

U0 的 encoder 对 mask 完全无关：forward 无条件编码全部四模态，mask 只在 temporal mask 与 pooling 阶段介入。因此
`latent_sequence` 与各 encoder 末层线性变换的输入（image/lidar 320 维、radar/gps 64 维，钩子位置由既有 layer manifest
审计确认）可以在冻结 U0 上一次性缓存，之后任意 mask 都能重算 reliability、unimodal logits、router 特征与融合输出。
选择缓存而非每个分支重跑 encoder，是因为四条路线由此共享逐比特相同的表征，arm 之间的差异只可能来自 router 输入 ——
这正是 BTMA 因 encoder 训练随机性而无法完成的归因隔离。缓存必须通过逐样本等价性测试：从缓存重算的融合 logits 与
直接跑冻结 U0 在同一 mask 下一致。

该等价性测试与整条缓存链路一律使用 float32，不使用既有评测路径的 bfloat16 autocast。理由是 bfloat16 不能充当参照：
同一冻结模型分别在 float32 与 bfloat16 autocast 下各跑一次，融合 logits 差异最大可达 1.5e-1，比缓存往返本身更大，
因此拿 bfloat16 前向当参照实际上是在比较两种不同的舍入，而两者都不是基准。改为 float32 之后，15 个 mask 上的
replay 与实时前向逐比特相同（最大差 0.0），等价性从“近似成立”变为“可精确验证”，代价只是缓存从每设定 474 MB 增至
947 MB。相应约束：本轮的绝对 Top-1 不与既有 bfloat16 A0 行直接可比，因此冻结 U0 参照行必须经同一条 float32 路径
重算，全部 arm 间比较都在该路径内部完成。

四条路线构成严格嵌套阶梯。Q0 只用现有五个标量与模态 one-hot，隔离“重训 router”本身的影响，其 Full 指标应落在冻结
U0 附近。Q1 追加每模态 prototype-space 状态（nearest distance、distance margin、entropy、restoration residual norm），
是关键对照：这些量完全位于投影下游，若 Q2 约等于 Q1，则“投影前信息”主张不成立。Q2 在 Q1 基础上追加投影前特征经小
MLP 得到的 quality embedding。Q3 与 Q2 参数量严格相等，但投影前特征在 batch 内跨样本置换，排除“只是多了参数”。

退化设定必须两个都跑。探针证据来自注入式腐蚀（45 条件，probe 仅 600 训练样本，标注 claim-ineligible），而 Full-pool
协议只有自然天气与硬 mask；只跑自然设定无法区分“机制不存在”与“该设定下没有可感知的退化”。设定 C 的 45 个条件按既有
diagnostic sample manifest 的 `corruption_type`/`severity` 重新实现，每个样本用固定种子预先抽定唯一条件后缓存。

预注册门槛在实现前固定：主指标为 Full-pool inner-validation 的 Top-1 与 all-14 mask 平均 Top-1，方向成立需同时满足
Q2 优于 Q1、Q2 优于 Q3，且 MAE 与 within-3 不劣化；三个 router seed 的区间不得重叠。任一不满足即判死，不得调参、加
seed 或访问 outer test。因为骨干已缓存，三个 router seed 近乎免费，直接补上 BTMA 上致命的“单 seed 不能声称显著”缺口；
但骨干本身仍是单 seed，必须在报告中写明。

处理组还必须执行冻结权重的推理期消融：评测时把每模态 quality embedding 换成训练集均值嵌入，直接读出该机制的推理期
贡献。这是本项目对新候选的通用准入测试 —— 推理期不存在的机制只能作为训练技巧，不能作为创新点。本轮只回答 routing
输入问题，不回答端到端联合训练问题。

## Risks / Trade-offs

- [所给 checkpoint 缺失或哈希不匹配] → 只列出带哈希候选，无法唯一确认立即退出，绝不回退到其他权重。
- [U0 原型只是分类 prototype 而非独立修复模块] → 审计清楚报告实际类型与推理用途；不虚构“修复残差”。
- [GPU1--7 任务资源不足或有无关进程] → 启动前只记录 `nvidia-smi`，不结束其他进程；作业 manifest 逐项记录失败原因。
- [Full 路径意外受到包装器影响] → 每个 Adapter run 在逐样本 Full logits 哈希和误差检查通过前标为失败。
- [NPZ 指标实现偏差] → 训练期汇总与独立 NumPy 重算逐项比较，最大绝对差超过 1e-7 即失败。
- [Stage A 偶然波动] → 只根据预注册门槛提出 Stage B 脚本，绝不自动运行多 seed。
- [历史 exclusion 明细源文件已清理] → 仅接受 source manifest 中 `excluded_development` block 的确定性恢复，并以每域计数和总计 588 双重 fail-closed 校验。
- [Full-pool 运行占用稀缺 GPU 过久] → 使用预注册、仅依赖训练损失的 min-8/patience-3/0.5% 相对改善早停；20 epoch 为上限，完整评估和审计不得省略。
- [FP16 prototype 梯度溢出或测时 resume 污染正式训练] → 使用 reference BF16 profile，将 probe 与正式 run 隔离，并在 Stage 2 前执行无标签 prototype 塌缩审计。
- [CUDA ordinal 指向错误物理卡] → 启动前解析并记录目标物理 GPU UUID，以 UUID 绑定子进程，且 fail-closed 拒绝未获授权的 GPU。
- [多核统计改变归一化或引入 validation] → 仅并行计算 train leaf 的局部 float64 moments，并按固定 domain 顺序归并；测试对照单线程结果与 train-only 元数据。
- [并行资源校验漏检重复窗口中的坏单元] → 必填单元和标签仍逐列向量化覆盖全部行，仅对已确认非空的实际资源路径去重；Radar 派生 `_DA` 也独立进入检查，并用缺失/非法路径回归测试保持 fail closed。
- [向量化 split audit 改变身份口径] → 保留 domain 前缀、完整行 canonical SHA256、history/future JSON 展开及全部资源列匹配，并以手工期望集合测试 sample、target、frame、trajectory 和资源身份。
- [旧缓存与当前变换不一致或缺失] → 正式读取严格校验参数、shape、source fingerprint 和覆盖率，任一 miss 都 fail closed；不允许训练时写缓存。
- [整样本 LMDB 重复滑窗数据耗尽磁盘] → 禁止复用旧 LMDB；仅在吞吐门槛失败后构建唯一帧分片 LMDB。
- [收尾统计被用来为 BTMA 翻案] → 门槛在计算前写入 OpenSpec，报告显式声明结论不可重开该路线；收尾不产生任何可调超参数。
- [把“不可区分”写成“显著劣于”] → 所有 BTMA 对比必须同时给出点差与 block bootstrap 区间，缺区间时只能表述为未超过对照。
- [缓存重算与真实 U0 前向不一致] → 逐样本等价性测试作为 Router 筛选的启动前置条件，不一致即 fail closed。
- [把 router 收益误归因于额外参数] → Q3 与 Q2 参数量断言严格相等，且 Q2 必须同时超过 Q1 与 Q3 才算通过。
- [用探针分数代替任务指标下结论] → 预注册主指标只有 Top-1 与 all-14，severity Spearman 仅作为机制诊断报告。
- [注入腐蚀被误当作部署协议结果] → 设定 C 与设定 N 分别独立报告，设定 C 结论明确标注为机制验证而非部署证据。
- [冻结骨干结论被外推为端到端结论] → 报告限定该筛选只回答 routing 输入问题，骨干仍为单 seed 且 encoder 未参与训练。
