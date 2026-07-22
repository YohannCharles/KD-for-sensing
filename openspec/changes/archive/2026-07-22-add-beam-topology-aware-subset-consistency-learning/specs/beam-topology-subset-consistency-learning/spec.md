## ADDED Requirements

### Requirement: 基础身份与 encoder-tail 插入点必须可审计
系统 MUST 从同一上游 encoder validation-best checkpoint 与 F1 feature concat MLP validation-best checkpoint 初始化，记录 checkpoint/config/cache SHA、prototype、temperature、topology、split 身份和每个模态最后投影输入的 module path 与维度。Stage B adapter MUST 作用于 image/lidar 320 维和 radar/gps 64 维 encoder-tail feature，且 MUST 位于 F1 256 维 token projection 之前。

#### Scenario: 校验在线与缓存插入点
- **WHEN** preflight 对固定 inner 样本同时执行在线 encoder 捕获和读取 Stage B cache
- **THEN** feature shape、sample identity、source SHA 与 F1 logits parity MUST 满足预注册容差
- **AND** 任一身份或 parity 不一致时所有训练 MUST 拒绝启动

### Requirement: Stage A specialist 必须探测原始输入信息上限
系统 MUST 在 GPU0--5 分别运行 Only Image、Only LiDAR、Only Radar、Only GPS、Image+Radar+GPS 和 Radar+GPS 六个固定 subset specialist。每个 specialist MUST 在线读取原始输入，解冻当前可用模态 encoder 与 F1 fusion/output projection，冻结 prototype bank，并只按对应 subset 的 inner-validation CE 加 topology loss 选择 checkpoint。

#### Scenario: 完成六个 specialist
- **WHEN** Stage A 的六个任务完成统一评测
- **THEN** 系统 MUST 报告各 subset 相对冻结 F1 的 Top1 headroom、Top3/Top5、Within-3、MAE、distance>5、weather、sector 与训练时间
- **AND** 报告 MUST 将其称为 specialist probe 而不是 oracle

### Requirement: 14-pattern schedule 与 nested pair 必须固定且完整
系统 MUST 精确枚举 4/6/4 个 single/double/triple missing pattern，拒绝 all-missing，并把 condition order 的四维 availability 映射为 F1 time-major 20-block mask。训练 MUST 先均衡 missing count 再均衡具体 pattern；V1--V5 MUST 共享 sample order、mask schedule、seed 和 batch order。系统 MUST 固定全部合法 `S subset T` pair，并优先从只多一个可用模态的 pair 采样。

#### Scenario: 生成训练与验证 manifest
- **WHEN** prepare 生成 train schedule、validation manifest 和 nested pair manifest
- **THEN** 三者 MUST 使用同一 sample identity、pattern id 和 availability/block-mask 映射
- **AND** 每个 nested pair MUST 非空、严格包含且来自同一原始样本

### Requirement: Canonical Full 与训练 Full 必须分别受控
V0 MUST 直接评测 canonical frozen F1，并在相同 sample IDs 上连续两次得到逐 pattern 完全一致的 Top1。V1--V5 MAY 更新预注册 encoder 尾部与 F1 fusion scope，但 MUST 对当前 Full 与 stop-gradient canonical Full teacher 计算 preserve KL，且 Full MUST 进入固定 checkpoint selection。

#### Scenario: 复现与训练 Full
- **WHEN** repair 运行 V0 两次 canonical 评测并训练 V1--V5
- **THEN** V0 两次 Full/14-pattern Top1 MUST 完全一致
- **AND** V1--V5 MUST 使用相同 preserve 实现、canonical teacher 和 Full validation loss，不得按 Full Top1 选择 checkpoint

### Requirement: 端到端尾部训练 scope 必须公平且受限
V1--V5 MUST 从同一 canonical F1 与 encoder checkpoint 初始化，统一解冻 image/lidar 的预注册最后 stage/block、radar/gps 的预注册最后 block、F1 modality adapter、modality/time embedding、fusion MLP 与 output projection；更早 encoder、prototype bank、64 prototypes 与 topology MUST 冻结。availability MUST 仅用于固定 subset mask，不得产生权重、reliability、gate 或 Router 输出。

#### Scenario: 执行固定 nested chain
- **WHEN** 同一 raw batch 构造 `S1 subset S2 subset S3 subset Full`
- **THEN** 每个 encoder MUST 只 forward 一次，四个 view MUST 在 fusion 前使用同一 token tensor 与固定 mask
- **AND** V1--V5 的 trainable parameter names、初始化 hash、optimizer、sample order 和 schedule hash MUST 一致

### Requirement: AER 必须只保留当前可用模态的独立证据
每个 modality auxiliary head MUST 只读取该模态 encoder-tail 的五时间步 feature，经 attention-free temporal pooling 和小 projection 使用冻结 prototype bank 产生 logits。AER MUST 只对当前可用模态计算 CE 加 topology loss，并先在每个样本内按可用模态平均；梯度 MUST 回传到对应 encoder adapter。auxiliary head MUST 可在部署时移除。

#### Scenario: 部分模态样本计算 AER
- **WHEN** V2/V5 样本只保留部分模态
- **THEN** 缺失模态 auxiliary head MUST 不被调用，其他模态 feature 与 availability 内容 MUST 不进入该模态 head
- **AND** auxiliary 改善只有在 fused 输出同时改善时 MAY 被判定为机制成立

### Requirement: NTM 必须使用现有 topology 且只优化 larger subset
NTM MUST 用现有 topology distance 计算预测风险，对同一样本的严格嵌套 subset 执行 `ReLU(R_T-stopgrad(R_S)+margin)`。少模态 risk、Full 参照和基础 F1 MUST 无梯度；系统 MUST 报告全部合法 pair 的 topology violation rate/magnitude、Top1、Within-3 与 MAE violation。

#### Scenario: 训练相邻 nested pair
- **WHEN** target subset S 与只多一个模态的 T 计算 NTM
- **THEN** `R_S` MUST detach 且 NTM 梯度只可流向 T 的新增 missing-path 参数
- **AND** T 为 Full 时 Full logits 与参数 MUST 保持冻结

### Requirement: SCFC 必须按缺失严重度聚合 topology sector
SCFC MUST 从固定 topology position manifest 构造 sector，并对 single/double/triple missing 分别使用 16/8/4-sector Full-teacher KL。Full teacher MUST stop-gradient，sector probability MUST 保持归一化，triple missing MUST NOT 使用 64-beam 精细 KL。

#### Scenario: 三种严重度计算一致性
- **WHEN** 同一 batch 分别包含 single、double 和 triple missing
- **THEN** 系统 MUST 分别选择 16、8 和 4-sector 聚合并使每行聚合概率和为一
- **AND** sector 构造 MUST 来自声明 topology 而不是未经确认的线性或首尾邻接假设

### Requirement: V0--V5 训练与 checkpoint 选择必须公平且预注册
V0 MUST 是不训练的 canonical frozen F1；V1--V5 MUST 使用相同结构、初始化、trainable scope、split、nested-chain schedule、optimizer、learning rate、epoch、batch order、seed、topology 和 prototype bank，只改变预注册的 AER/NTM/SCFC loss，V5 另使用固定 `1:1:1:1:4` 日程。所有 lambda MUST 只用固定 train batches 校准一次；checkpoint MUST 只按 `0.25 * (L_full + L_single + L_double + L_triple)` 选择。

#### Scenario: 比较五个可训练方向
- **WHEN** V1--V5 完成一个 validation epoch
- **THEN** Top1、All-14、Worst、Radar+GPS、monotonicity、weather 和 sector MUST NOT 影响 checkpoint 或 lambda
- **AND** resolved config MUST 证明除 loss 组件外结构与公共训练设置一致

### Requirement: Repair 必须锁定 canonical 身份并解释历史漂移
repair MUST 保留旧结果，先审计 V0--V5 的日志/PID/status/checkpoint/metrics 并使用受限状态枚举。canonical F1 MUST 是 Availability Fallback U0 resolved config 直接引用的 checkpoint，而不是按指标选择；系统 MUST 保存 checkpoint/config/prototype/topology/metric 与 train/validation/development sample hash，并在统一协议下重评六个 specialist。

#### Scenario: specialist baseline 与训练对象隔离
- **WHEN** repair 加载既有 specialist checkpoint 完成同 subset 重评
- **THEN** baseline MUST 来自独立重新加载且未被 specialist optimizer 更新的 canonical F1 对象
- **AND** 报告 MUST 用 checkpoint、sample、mask、normalization、metric 与 split 证据解释历史 baseline 差异，不得归因于随机误差

### Requirement: 统一评测、GPU 隔离与停止边界必须完整
系统 MUST 对 Stage A 与 V0--V5 生成逐 pattern、missing-count macro/worst、modality-absent、nested chain、monotonicity、error-distance、weather、sector、representation/probe 和效率结果，并按七项 gate 给出唯一推荐方向。GPU0--5 子任务 MUST 保存 PID、日志、resolved config、状态和退出码；一个任务失败 MUST NOT 终止其他任务。所有结果 MUST 标记 single-seed、inner/development、claim-ineligible，且 MUST NOT 自动调参、重跑、outer test、multi-seed 或下一轮。

#### Scenario: 生成最终 comparison
- **WHEN** 所有可用 Stage A 与 Stage B 结果完成汇总
- **THEN** 系统 MUST 生成预注册 CSV/JSON/Markdown、逐项 success gate、失败任务说明和唯一推荐方向
- **AND** 缺失结果 MUST 保持缺失或 failed，系统 MUST NOT 伪造指标或自动扩大实验范围

#### Scenario: Repair smoke 与正式矩阵隔离
- **WHEN** repair 准备在 GPU0--5 启动正式 V0--V5
- **THEN** 每个配置 MUST 先在对应 physical GPU 经 `CUDA_VISIBLE_DEVICES` 映射到内部 `cuda:0`，通过 2 train steps、1 validation batch、checkpoint save/load 与 metrics write smoke
- **AND** repair MUST 写入独立 `outputs/bt_scl_repair/`，保留旧目录并分别记录六个正式任务的 PID、退出码、checkpoint、metrics 与 final status
