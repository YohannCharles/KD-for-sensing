## Context

F1 由四个模态 encoder 的最后投影输入特征（image/lidar 为 320 维，radar/gps 为 64 维）构成 20 个 pre-prototype block，经 `FeatureTokenAdapter` 投影到 256 维后使用 concat MLP 与冻结 prototype bank 预测。上一轮只在 256 维 F1 token 后适配，未形成稳定收益；本轮需要把可训练增量前移到 encoder 最后投影之前，同时保持 Full 对原 F1 的物理直通。

本轮只使用既有 inner train/validation/development 身份、四模态五时间步、beam label 与 checkpoint 声明的 64-beam topology。Stage A 用原始输入探测信息上限；Stage B 基础 encoder 与 F1 全部冻结，因此可复用已审计的 pre-prototype cache 作为 encoder-tail 插入点输入，而无需重复执行冻结 backbone。

## Goals / Non-Goals

**Goals:**

- 以六个在线 subset specialist 判断原始输入与 encoder 是否存在 F1 未利用的信息。
- 以同一组 encoder-tail 和 fusion residual 公平比较 V1--V5，仅改变 AER、NTM、SCFC loss 组合。
- 对 Full 保证逐元素一致、adapter forward count 为零，并完整覆盖 14 个非空缺失 pattern 与全部嵌套 pair。
- 生成可审计的 pattern、单调性、表示、天气、sector、错误距离、效率与 success-gate 证据。

**Non-Goals:**

- 不实现动态 Router、模态权重、quality/reliability、attention/MoE、logit 加权创新、重建或 residual recovery。
- 不读取 channel、CSI、gain、power、ray/path、历史 beam、weather、scene 或 beam label 作为模型输入。
- 不新增公共 CLI、canonical recipe，不自动运行 outer test、multi-seed 或下一轮。

## Decisions

### 1. Stage A 在线训练，Stage B 使用等价的 encoder-tail cache

Stage A 从上游 C0 validation-best checkpoint 构建四个原始 encoder，并在每个 encoder 最后一层线性投影之前捕获特征；specialist 解冻当前可用 encoder 的有效前缀、F1 token/fusion/output projection，prototype bank 保持冻结。这样 probe 能回答原始输入经 encoder fine-tuning 后的 headroom，而不是只回答冻结 token 的线性可分性。

Stage B 在完全冻结 encoder 的前提下直接读取同一插入点的 `[B,5,4,Dmax]` cache，并在 320/64 维特征上施加低秩 residual。对固定 encoder，缓存输入与在线插入得到相同梯度和输出；该位置早于 F1 的 256 维 token projection，因此不复用上一轮 frozen-token adapter。备选方案是每个 epoch 重算冻结 TinyViT，计算成本更高且不会改变可训练函数，故不采用。

### 2. 一个确定性 regime 模型覆盖 V1--V5

共享模型包含 Regime B 的四个 modality-specific low-rank residual、Regime C 的 radar/gps low-rank residual、两个 availability-conditioned fusion residual 和四个训练期 auxiliary prototype heads。所有 residual 的末层零初始化。Regime A 不调用新增 adapter；Regime B/C 只调用当前可用模态的确定分支。availability 只参与布尔路径选择、固定 embedding 和一致性粒度，不产生权重或概率。

V1--V5 实例化完全相同的模块与初始化；V1 仅用 task loss，V2/V3/V4 分别增加 AER/NTM/SCFC，V5 同时增加三者。V0 直接评测冻结 F1。

### 3. Full 在任何新增模块前返回

当四维 availability 全为真时，forward 直接把原 pre-prototype feature 与全可用 block mask交给冻结 F1，不构造 regime embedding、不调用 encoder/fusion residual 或 auxiliary head。混合 batch 中的 Full 行也只保留基础 F1 输出。测试以同一 tensor 调用原 F1 和 wrapper，要求 `torch.equal`，并检查所有新增模块计数为零。

### 4. 固定 schedule、nested pair 与 topology sector

复用现有 14-pattern 枚举和 condition-to-time-major mask 映射。训练对 missing count 1/2/3 等预算分配，再在组内轮转 pattern，并冻结每 epoch sample order。每个 target subset 的 larger subset 从多一个可用模态的相邻候选中按 sample identity、epoch 和固定 seed 选择；验证枚举全部合法 `S subset T` pair。

距离矩阵只由 checkpoint/cache manifest 中声明的 topology id、permutation 和现有 `beam_topology_distance_matrix` 构造。sector 按 topology position 聚合为 16/8/4 组，不从 beam label 数值猜测邻接关系；manifest 固定 label 到 sector 的映射。

### 5. Loss 与 checkpoint 选择预先固定

Task loss 为 CE 加既有 soft topology loss。AER 对当前可用模态先逐样本平均；NTM 对少模态 risk stop-gradient；SCFC 对 single/double/triple 分别使用 16/8/4-sector Full-teacher KL，Full teacher stop-gradient。所有 lambda 只在固定 train batches 上按目标量级校准一次，V1--V5 共用结果。

checkpoint selection 只使用 single/double/triple validation task-loss macro 的平均；V3/V5 额外加入预注册的固定比例 validation NTM。Top1、Worst、weather、sector 和 outer test 不参与选择。

### 6. 本地 analysis workflow 承担训练与汇总

实现保留为非 registry 模型/loss 组件和一个 analysis 入口；两个 shell 脚本仅负责 GPU0--5 的独立子进程、PID、日志与退出码。每个任务失败不终止其他任务，汇总只消费实际存在且身份匹配的结果。

## Risks / Trade-offs

- [Risk] Stage B cache 被误解为 token-level 适配。 -> manifest 明确记录捕获的 encoder module path、维度和 source SHA；测试验证 adapter 位于 F1 token projection 之前。
- [Risk] Stage A 在线特征与 Stage B cache 漂移。 -> 启动前在固定样本比较在线/缓存 feature 与 F1 logits，未通过 parity gate 时拒绝训练。
- [Risk] NTM 通过恶化少模态分支降低违反。 -> `R_S` 必须 detach，只有 larger branch 接收该项梯度。
- [Risk] topology/sector 错配导致伪一致性。 -> 只接受 manifest 支持的 topology，保存距离与 sector mapping checksum，并测试概率聚合和为一。
- [Risk] 多分支 loss 量级失衡。 -> 固定 train batch 一次性校准并保存原始/加权比例，禁止 validation 后调参。
- [Risk] Stage A 全 encoder fine-tuning 成本较高。 -> 只运行六个预注册 specialist、单 seed 和 early stopping；不扩展矩阵。

## Reproducibility Repair Revision

### 7. 先审计，再锁定 U0 直接引用的 canonical F1

旧 V0--V5 结果缺失必须先按日志、PID、status、checkpoint 和 metrics 证据分类，不把 `never_launched` 写成训练失败。repair 只选择 Availability Fallback U0 resolved config 直接引用的 F1 checkpoint，不按性能挑选。canonical manifest 同时锁定 config/prototype/topology/normalization/metric 与 train/validation/development sample hash；同一固定 development IDs 连续评测两次，逐 pattern Top1 必须完全一致，否则停止正式矩阵。

### 8. Specialist 与 canonical baseline 使用独立模型对象

旧 Stage A 在 specialist 训练后使用同一个已更新的 F1 fusion 对象计算 baseline，导致 weak-only baseline 漂移。repair 优先复用六个既有 validation-best specialist checkpoint，但重新独立加载 canonical F1，并在同一 development IDs、mask、topology 和 metric 实现下重算 baseline 与 specialist。checkpoint/config/subset/sample 身份不一致时该 specialist 标记无效并仅按原预注册 recipe 重训。

### 9. Repair Stage B 使用统一端到端尾部训练 scope

V0 仅评测 canonical frozen F1。V1--V5 从同一 canonical F1 与上游 encoder 初始化，统一解冻四个 encoder 的预注册最后 stage/block、F1 modality adapter、modality/time embedding、fusion MLP 与 output projection；冻结更早 encoder、prototype bank、64 prototypes 与 topology。每个 raw batch 的 encoder 只 forward 一次，再构造固定 `S1 subset S2 subset S3 subset Full` 四个 fusion views，共享 encoder 梯度。

V1 的公共目标为四个 view 的 task loss与 canonical Full preserve KL；V2 增加对应模态 auxiliary prototype loss，V3 增加相邻 view topology monotonicity，V4 增加 4/8/16-sector 和 single-missing local consistency，V5 使用固定 `1:1:1:1:4` round-robin auxiliary/joint 日程。相同 loss 的 lambda 只在固定 train batches 校准一次并跨方法共享。

### 10. Full 不再物理 bypass，但受固定 preserve 与 selection 约束

repair 的端到端 scope 会更新当前 Full 路径的尾部参数，因此旧的逐元素 bypass 只作为 V0 canonical 参考，不再作为 V1--V5 结构要求。V1--V5 以 stop-gradient canonical Full teacher 计算 preserve KL，并统一用 `0.25 * (L_full + L_single + L_double + L_triple)` 选择 checkpoint；Top1、Worst、Radar+GPS、weather、sector 和 monotonicity 不参与选择。

### 11. 新目录隔离 repair 证据

旧 `outputs/bt_subset_consistency/` 不覆盖、不删除。`outputs/bt_scl_repair/legacy_snapshot/` 保存报告、日志/config/specialist 清单与 SHA，repair 的 audit、canonical manifests、smoke、正式运行和汇总全部写入新目录。GPU 子进程只接收一个 physical GPU，经 `CUDA_VISIBLE_DEVICES` 映射后内部统一使用 `cuda:0`；失败任务不影响 sibling，也不自动重跑。
