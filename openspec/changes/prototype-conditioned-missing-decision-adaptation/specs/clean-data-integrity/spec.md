## ADDED Requirements

### Requirement: Adapter Stage A 遵守 clean inner protocol
Prototype-conditioned decision Adapter 训练 MUST 在构建 U0、Adapter、optimizer 或 loader 前重新验证 `mmw_clean_inner_development_v1` protocol 与审计报告。它 MUST 仅使用 inner_train 优化 20 个 epoch，仅使用 inner_validation 进行只读 15-mask 评估，并将 `outer_test_accessed=false` 写入运行审计。

#### Scenario: 协议或隔离审计无效
- **WHEN** protocol/audit 缺失、不匹配、outer test 被启用，或任何 sample id、target id、完整 CSV 行、原始输入、window frame、target frame、sequence/trajectory 存在 train-validation 重叠
- **THEN** 工作流 MUST 在访问训练数据或创建 optimizer 前失败

### Requirement: Adapter 的可拟合状态仅来自 inner_train
Adapter 参数、AdamW 状态、cosine scheduler、条件标准化统计、mask schedule 和 A7 train permutation MUST 仅由 inner_train 及固定随机种子构建。inner_validation 不得参与早停、rank/loss 选择、checkpoint 选择、条件归一化拟合或原型更新。

#### Scenario: 评估期间状态不可变
- **WHEN** Stage A 对 inner_validation 执行任意 mask 的评估
- **THEN** U0、prototype、Adapter 参数、条件归一化统计、optimizer 和 scheduler 状态 MUST 保持不变

### Requirement: Full-pool capacity protocol 使用共享连续时间轴并保持 outer 锁定
Full-pool capacity workflow MUST 从 15 个 Town3 `all_sequences.csv` 的 46,860 个候选窗口构建 `mmw_full_pool_development_v1`。当按 CAV 定义的 trajectory/session 共享 RSU Radar 或 BS-GPS 资源时，系统 MUST 拒绝将该标识视为可靠 group split，并改用每个 domain 全部 CAV 对齐的单一 80/20 连续时间边界；只有依赖集合跨越真实边界的窗口可被 purge。

#### Scenario: 构建 Full-pool 开发划分
- **WHEN** 15 个候选源及其哈希、帧依赖和增强资源都有效
- **THEN** train 与 inner validation MUST 均覆盖 15 个 domain
- **AND** sample、target、完整 CSV 行、window/target frame、camera、lidar、radar、UE GPS、BS GPS 和 channel/beam 资源交集 MUST 全为零
- **AND** 任一交集、缺失增强资源或无效行未显式报告时 MUST 在训练前失败

### Requirement: 历史验证身份排除可验证且只约束训练
Full-pool protocol MUST 从受哈希约束的历史 source manifest 中 `excluded_development` block 恢复历史身份，并逐 domain 对齐 `excluded_window_count`，总数 MUST 严格为 588。恢复过程 MUST 不读取 outer-evidence CSV 的样本、标签、统计量或结果；恢复出的身份 MUST 不进入新 train。

#### Scenario: 历史排除证据不完整
- **WHEN** source manifest、source CSV 或计数校验缺失或不一致
- **THEN** protocol construction MUST fail closed
- **AND** 系统不得以忽略排除、读取 outer evidence 或猜测身份作为回退

### Requirement: Full-pool 可拟合状态只来自新 train
Radar/BS-GPS 路径增强 MUST 由原始 scene 共享时间轴确定性生成并在增强后重新审计。GPS scaler、normalization、频率先验、prototype 和 Adapter condition normalizer MUST 只由 Full-pool train 拟合；inner validation 只用于训练完成后的固定 15-mask 评估，`outer_test_accessed` MUST 保持 false。

Full-pool pooled dataset 与 GPS scaler MAY 使用多核并行构建，但 GPS 局部统计 MUST 只读取 train leaf，并按固定 domain 顺序确定性归并。prepared CSV 资源校验 MAY 在逐单元必填检查后按真实资源路径去重并并行执行，但 MUST 仍覆盖所有启用模态、Radar `_DA` 派生资源和 Beam 标签，任一非法或缺失值 MUST fail closed；pooled 构造的资源校验总并发 MUST 不超过 90。并行与单线程的 sample/frame count MUST 一致，且 validation 不得进入任何局部统计任务。

Full-pool split audit MAY 使用逐列向量化集合构造代替逐行全列扫描，但 MUST 保留 domain-scoped sample、target、window/target/all-frame、trajectory/session、完整 CSV 行 SHA256 及所有资源族的相同身份口径；任一 train-validation 交叉仍 MUST fail closed。

#### Scenario: Full-pool validation 运行
- **WHEN** U0 或 Adapter 对 inner validation 枚举 mask
- **THEN** validation MUST 不参与逐 epoch 早停、checkpoint 选择、rank/loss/lr 调整或任何状态更新

### Requirement: Full-pool 派生缓存复用必须可验证且只读
Full-pool workflow MAY 复用由候选池原始资源确定性生成的图像、LiDAR 与 GPS 输入缓存，但 MUST 绑定 source CSV hash、protocol fingerprint、变换代码 hash、参数和资源覆盖率。正式训练的图像与 LiDAR cache miss、metadata mismatch 或 shape mismatch MUST fail closed，不得静默回退或由 DataLoader worker 写缓存。GPS scaler MUST 继续只由 Full-pool train 拟合；validation GPS 坐标即使被无标签缓存，也不得进入 scaler moments。

#### Scenario: 正式 DataLoader 复用现有帧缓存
- **WHEN** Full-pool U0 或 Adapter 加载缓存资源
- **THEN** train/validation 引用的每个图像和 LiDAR 资源 MUST 命中参数一致的缓存
- **AND** 缓存张量 MUST 与当前原始变换数值一致
- **AND** outer test MUST 保持未访问

#### Scenario: 复用 train-only GPS scaler
- **WHEN** 正式 U0 从测时 probe 加载 GPS scaler
- **THEN** artifact sample identity hash、sample count、feature mode 与 protocol train role MUST 完全一致
- **AND** 任一 provenance 或 SHA256 不一致 MUST 在构建 optimizer 前失败

### Requirement: ADBA-surrogate follow-up 复用同一隔离边界
B1/B4/B6/B7 MUST 复用已审计的 Full-pool train、inner validation、train-only normalization、mask schedule 与唯一 U0 checkpoint。inner validation MUST 只用于训练结束后的固定 15-mask 评估，不得参与 surrogate 权重、soft-label sigma、早停或 checkpoint 选择。

#### Scenario: B 组启动前审计
- **WHEN** 任一 B1/B4/B6/B7 子任务启动
- **THEN** protocol fingerprint、schedule SHA256 与 U0 SHA256 MUST 与 A0--A7 相同
- **AND** `outer_test_accessed` MUST 保持 false

### Requirement: Unseen-mask pilot 只排除输入视图而不改变物理样本边界
Mask-bias novelty triage MUST 复用同一 Full-pool train、inner validation、train-only normalization 与唯一 U0。Unseen fold 只能从 train mask schedule 排除 held-out mask，不得移动、增加或删除物理 train/validation 样本；inner validation 仍只在训练结束后读取。

#### Scenario: 生成 held-out schedule
- **WHEN** fold 0 schedule 被生成
- **THEN** train sample identity 集合与原 schedule MUST 完全相同
- **AND** held-out mask 曝光数 MUST 为零，允许 mask 的曝光次数 MUST 在每 epoch 尽量均衡
- **AND** outer test MUST 保持未访问

### Requirement: 环形传输比较复用 Full-pool 训练边界
Circular Transport 与 all-seen Factorized Bias 比较 MUST 复用同一 Full-pool train、inner validation、train-only normalization、唯一 U0 和完整 14-mask train schedule。它不得基于 validation 选择核半径、训练 epoch、损失权重或 checkpoint。

#### Scenario: 启动环形传输对照
- **WHEN** 编排器启动 Circular Transport 或 all-seen Factorized Bias
- **THEN** 两者 MUST 读取完全相同的 train sample identity、mask schedule SHA256 和 U0 SHA256
- **AND** outer test MUST 保持未访问

### Requirement: BT-SCL 严格复用 Full-pool 且 validation 只读
Full-Pool BT-SCL MUST 绑定 `mmw_full_pool_development_v1` 的 exact manifest/audit，使用 37,038 train 与 9,180 validation，并在创建模型、optimizer 或 loader 前复核 candidate=46,860、boundary purge=240、train historical removal=402 与 validation historical retention=186。它不得读取 clean-inner、outer evidence、outer test、channel 或 path；GPS 和所有 normalization state MUST 只由 Full-pool train 拟合。

#### Scenario: BT-SCL 预检失败
- **WHEN** protocol identity、资源零交集、训练/验证计数、outer flag、legacy-source flag 或 input contract 任一不满足
- **THEN** 系统 MUST 在启动训练前失败并写出审计原因

### Requirement: R6 不改变 BT-SCL 数据边界
R6 MUST 复用同一个 BT-SCL protocol fingerprint、37,038 train、9,180 validation、train-only GPS scaler、nested schedule 和公共初始化。R6 的 hierarchy、dominance 权重与半径 MUST 在读取 validation 前固定；outer test、clean-inner、channel 和 path MUST 保持未访问。

#### Scenario: R6 启动
- **WHEN** R6 创建模型或 optimizer
- **THEN** protocol、topology、normalization、schedule 与初始化哈希 MUST 与既有 R0--R5 完全一致
- **AND** 任一不一致 MUST fail closed

#### Scenario: R6 stable follow-up 使用 inner validation 早停
- **WHEN** post-hoc stable R0/R3/R6 从 epoch 6 起检查统一 selection loss
- **THEN** validation MUST 只读且只控制相同 patience/最佳 checkpoint，不得修改 loss、半径、sector、rank、数据、normalization 或 optimizer profile
- **AND** outer test MUST 保持未访问，运行状态 MUST 明确记录 `claim_eligible=false`

### Requirement: Candidate12 严格复用 Full-pool 且所有分配状态只来自 train
Full-Pool Candidate12 MUST 在创建 warm-up、模型或 optimizer 前复核 46,860 candidate、240 boundary purge、402 train historical removal、186 validation historical retention、37,038 train、9,180 validation及所有资源零交叉。它 MUST 不读取 clean-inner、outer evidence、outer test、channel/path/beam power 或历史 beam；GPS normalization、KL/risk percentile、容量分配、head-tail frequency 与 mean motion MUST 仅由 train 拟合。

#### Scenario: Candidate12 训练或 assignment 更新
- **WHEN** warm-up、A0--A5 或任一两 epoch assignment 更新启动
- **THEN** 物理 train/validation identity MUST 保持不变且 validation 只可用于 Full checkpoint selection 和训练结束后的诊断
- **AND** outer test MUST 保持未访问，任一协议、资源或 train-only provenance 不一致 MUST fail closed

### Requirement: BTMA 收尾只消费既有产物且不扩大数据访问
BTMA 收尾 MUST 只读取已发布的 BTMA checkpoint、既有 Full-pool validation loader、既有 assignment CSV 与既有 warm-up train prediction cache。它 MUST NOT 读取 outer test、channel、path、beam power、历史 beam 或未来 GPS，MUST NOT 拟合任何新的 normalization 或可学习状态。

#### Scenario: 收尾重算逐样本预测
- **WHEN** 收尾工具加载任一 BTMA checkpoint 并遍历 validation
- **THEN** 模型 MUST 处于 eval 与 no-grad，且 sample identity MUST 与既有协议一致
- **AND** outer test MUST 保持未访问，重算 MUST NOT 写回或覆盖原 BTMA 运行产物

### Requirement: Router 筛选的全部可拟合状态只来自 Full-pool train
Router 可观测性筛选 MUST 绑定与冻结 U0 相同的 Full-pool protocol fingerprint、37,038 train 与 9,180 validation。router、quality 分支、任何输入标准化统计、腐蚀条件抽取种子与推理期消融所用的均值嵌入 MUST 仅由 train 拟合或生成。validation MUST 只读，MUST NOT 参与早停、超参数选择、条件抽取或 checkpoint 选择。

#### Scenario: 构建冻结表征缓存
- **WHEN** 系统为设定 N 或设定 C 构建缓存
- **THEN** train 与 validation MUST 使用同一冻结 U0 与同一确定性顺序，且 U0 MUST 保持 eval
- **AND** 每个样本的腐蚀条件 MUST 由固定种子在读取任何指标前抽定
- **AND** outer test、clean-inner、channel 与 path MUST 保持未访问

#### Scenario: 推理期消融使用均值嵌入
- **WHEN** 处理组执行冻结权重推理期消融
- **THEN** 替换所用的均值嵌入 MUST 只由 train 计算
- **AND** 消融 MUST NOT 更新任何参数，MUST NOT 依据 validation 结果选择替换目标
