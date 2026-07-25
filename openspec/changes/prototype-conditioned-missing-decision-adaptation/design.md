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

Full-pool pooled dataset 的 15 个 domain 构造使用标准库线程池并行。每个 domain 的 prepared CSV 先向量化检查必填资源单元、Radar `_RA` 约束与 Beam 标签，再按真实资源路径去重；只有去重后的 camera/lidar/radar/UE GPS/BS GPS（含 Radar `_DA`）路径进入并行 `is_file` 校验。15 个 domain 各最多使用 6 个资源校验 worker，使 pooled 构造的总资源校验并发不超过 90；任何缺失、非法或越界单元仍按首个确定性位置 fail closed。train-only GPS scaler 将每个 train leaf 的 float64 sum/sum-of-squares 独立并行计算，再严格按 manifest domain 顺序归并。这样只改变吞吐，不改变 train-only 边界、样本集合、确定性顺序或 validation 只读契约。

Full-pool split audit 不再对每行重复扫描全部列。sample/target/trajectory 与各资源族使用逐列向量化集合构造，历史/目标 frame JSON 和完整行 SHA256 使用 `itertuples` 保持逐样本覆盖；输出仍是相同的强哈希身份集合及 train-validation 零交叉判据。

### Full-pool 复用确定性帧缓存并将 LMDB 置于测量门槛之后

现有 `outputs/cache/MMW/<condition>/image_derived` 与 `lidar_bev` 已由 15 个 `all_sequences.csv` 的原始资源生成。正式工作流只在以下条件全部满足时以只读方式复用：新 train/validation 引用资源覆盖率为 100%、缓存参数与当前 ImageNet RGB/LiDAR BEV 变换一致、代表性逐元素重算误差为 0、缓存复用 manifest 绑定 source CSV、Full-pool protocol fingerprint 与当前变换代码 SHA256。正式 DataLoader 对 cache miss、metadata mismatch 或 shape mismatch fail closed，不得静默回退到原始转换。

GPS 原始坐标属于无标签确定性输入，可按 domain 预先保存为紧凑 NPZ 并由 DataLoader worker 只读加载。GPS scaler 仍只由 Full-pool train 拟合；测时 probe 产生的 scaler 只有在其 metadata、sample identity hash、sample count、artifact SHA256 与同一 Full-pool protocol 完全一致时才能被正式 U0 复用。validation 坐标可被缓存，但不得进入 scaler moments。

旧 split-level LMDB 绑定历史 split、缺少 Radar 且按重叠窗口重复完整五帧样本，因此不得用于本轮。先测量严格 NPY cache DataLoader 吞吐；只有其稳定吞吐低于已测 U0 训练消费速率的 1.5 倍时，才允许构建按 15 个 domain 分片、只存唯一帧且不包含标签或归一化状态的新 LMDB。该门槛通过时必须跳过 LMDB，避免无证据复制约 54GB 派生张量。

测时 epoch 只用于估计吞吐，写入独立 `u0_timing_probe`，不得作为正式 U0 的 epoch 1。动态预算确定后，`u0_seed1` 从相同 seed 的随机初始化重新训练，以免 1-epoch probe 的 optimizer、AMP 或 scheduler 终态污染正式轨迹。Full-data 派生配置显式固定旧 3,600 clean U0 的对照 profile：关闭 router pattern bias、BF16 无 GradScaler、AdamW、weight decay 3e-4、cosine warm restart、temporal-missing seed 1；不从本地旧 checkpoint 或 outputs 读取配置。

正式 checkpoint 发布前计算归一化 prototype 的非对角两两余弦统计。旧 clean U0 的均值约为 -0.01，而失败 run 为 0.99956，因此以 0.95 作为仅检测明显塌缩的 fail-closed 阈值。该检查不读取 validation，也不用于选择 checkpoint 或调参。

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
