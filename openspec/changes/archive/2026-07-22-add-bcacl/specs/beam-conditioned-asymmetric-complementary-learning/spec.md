## ADDED Requirements

### Requirement: BCACL 默认关闭并保持基线等价
系统 MUST 提供 `bcacl.enabled` 开关且默认值为 false。关闭时系统 MUST 不实例化 BCACL 参数或 buffer，不改变模型 state dict、损失、训练 forward、推理融合、数据处理、随机种子、checkpoint 选择或第一创新点的融合恢复原型逻辑。

#### Scenario: 关闭 BCACL 加载既有 checkpoint
- **WHEN** 同一既有 checkpoint 和输入在未声明 BCACL 或声明 `bcacl.enabled=false` 的配置下运行
- **THEN** state-dict key MUST 与变更前一致
- **AND** 输出 MUST 在既有数值容差内一致

### Requirement: 单模态监督使用独立投影与 observed mask
启用 BCACL 时，系统 MUST 为每个配置模态建立独立的简单投影层和可选私有 64 类头，并 MUST 提供所有模态共享参数的可选 64 类线性头。单模态损失 MUST 只使用自然观测的 `observed_mask`，不得把 synthetic dropout 后的 `fusion_mask` 当成自然缺失。

#### Scenario: 人工屏蔽教师模态
- **WHEN** 一个自然观测模态在训练期被 synthetic dropout 从 fusion mask 中移除且 `distill_from_pre_dropout_modalities=true`
- **THEN** 该模态 MUST 仍可参与 Phase 1 单模态监督与 BCACL 教师资格
- **AND** 该模态 MUST 不进入当前样本的融合分支

#### Scenario: 自然缺失模态
- **WHEN** 数据基准可用性表明某模态原本不存在
- **THEN** 系统 MUST 不生成该模态特征、不计算其单模态损失且不允许其参与迁移

### Requirement: 模态原型与融合恢复原型严格分离
系统 MUST 维护形状为 `[M,K,D]` 的模态 Beam 原型库和形状为 `[M,K]` 的初始化、样本计数与质量状态，并 MUST 使用与现有融合恢复原型不同的名称、buffer 和更新路径。模态原型 MUST 只由训练集自然观测特征以 float32 统计，DDP 下 MUST all-reduce，样本不足的类别 MUST 保持上一轮状态。

#### Scenario: epoch 原型更新
- **WHEN** 一个训练 epoch 结束且某模态某 Beam 的观测样本数达到 `min_class_count`
- **THEN** 系统 MUST 按配置使用 epoch 均值替换或 EMA 更新对应 L2 归一化原型
- **AND** validation/test 样本 MUST 不参与统计

#### Scenario: 零样本类别
- **WHEN** 某模态某 Beam 在当前训练 epoch 没有足够样本
- **THEN** 该原型 MUST 保持旧值且 Beam 关系计算不得产生 NaN 或 Inf

### Requirement: 知识迁移发生在 Beam 关系空间并停止教师梯度
系统 MUST 对归一化投影特征和各自模态原型计算 64 维 `log_softmax` Beam 关系分布。迁移 MUST 使用 `KL(teacher.detach() || student)`，只更新学生分支；系统 MUST 不以原始异构特征 MSE、cosine 或全特征双向对齐作为 BCACL 目标。

#### Scenario: 教师与学生反向传播
- **WHEN** toy batch 只对一项 BCACL KL 执行 backward
- **THEN** 教师特征由该损失产生的梯度 MUST 为零或不存在
- **AND** 学生特征梯度 MUST 非零

#### Scenario: 没有有效迁移
- **WHEN** batch 中没有教师与学生同时满足观测、初始化和质量条件
- **THEN** BCACL loss MUST 是连接计算图、设备和 dtype 正确的零 tensor

### Requirement: 固定教师与质量教师均为稀疏单向关系
固定模式 MUST 从配置中的 dataset identity 到模态名映射解析教师，不得在 loss 中散落硬编码。质量模式 MUST 仅使用训练集原型和真实 Beam 类别统计，从其他 observed 模态中为每个学生每个样本最多选择一个质量最高教师，并只在质量超过学生至少 margin 时迁移。

#### Scenario: 固定教师数据集映射
- **WHEN** dataset type 与 `fixed_teacher` 的一个规范化 key 匹配
- **THEN** 系统 MUST 仅从该配置模态向同样 observed 的其他模态迁移
- **AND** 教师本身 MUST 不成为学生

#### Scenario: 自动教师过滤
- **WHEN**候选教师未初始化、样本数不足、与学生相同或质量差不超过 margin
- **THEN** 该候选 MUST 不产生迁移项
- **AND** 每个学生样本的有效教师数 MUST 不超过一

### Requirement: 原型质量采用训练集类内方差与 hard-negative 分离度
系统 MUST 维护 `[M,K]` 质量矩阵，在 warmup 后按 interval 更新并支持 EMA。默认 hard negative MUST 是当前模态原型空间中最近的有效其他 Beam 原型；在没有可审计码本物理映射时 MUST 不默认假设 Beam 序号环形邻接。

#### Scenario: 更新质量矩阵
- **WHEN** Phase 1 到达 warmup 后的质量更新 epoch
- **THEN** 系统 MUST 由训练 epoch 的归一化特征统计计算类内方差和最近原型分离度
- **AND** 低样本或未初始化类别 MUST 标记为不可作为教师

### Requirement: BCACL 支持独立消融
系统 MUST 支持 U0 当前基线、U1 仅私有监督、U2 私有加共享监督、U3 固定教师 Beam 关系蒸馏、U4 自动质量教师和 U5 detached two-stage，不得把这些变化绑定为不可分离的单一开关。

#### Scenario: 关闭共享头与迁移
- **WHEN** 私有头开启、共享头关闭且 `lambda_bcacl=0`
- **THEN** Phase 1 MUST 只由有效私有 CE 更新对应编码器与私有分支

### Requirement: BCACL 不改变推理融合
BCACL 的投影、私有头、共享头、模态原型和质量矩阵 MUST 只服务训练与诊断，推理输出 MUST 继续由当前缺失模态融合和第一创新点融合恢复原型产生。

#### Scenario: Phase 2 或评估 forward
- **WHEN** 运行 Phase 2 融合训练或任意评估
- **THEN** 系统 MUST 不使用 BCACL 质量、教师或关系分布作为融合权重

### Requirement: BCACL 诊断持久化
系统 MUST 每个 Phase 1 epoch 将单模态性能、私有/共享 CE、BCACL loss、有效/跳过迁移数、4x4 教师学生计数、逐 Beam 教师分布、样本计数、原型初始化率和平均质量保存为 JSON/CSV，并将适用标量写入现有日志系统。

#### Scenario: 写出教师矩阵
- **WHEN** 一个 Phase 1 epoch 结束
- **THEN** 持久化的 4x4 迁移矩阵对角线 MUST 为零
- **AND** 模态顺序、dataset、epoch 与 stage identity MUST 一并记录

### Requirement: BCACL 实验保持 inner-only
BCACL smoke、消融、固定教师与自动教师运行 MUST 为 single-seed、inner train/validation/development 且 `claim_eligible=false`，不得自动触发 outer test、multi-seed、正式超参数选择或 claim 更新。

#### Scenario: 夜间实验完成
- **WHEN** 本地 BCACL 夜间任务完成
- **THEN** 系统 MUST 只保留本地 checkpoint、日志和 development 汇总
- **AND** 不得自动运行 outer test 或修改正式 claim
