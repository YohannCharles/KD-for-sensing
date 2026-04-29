## MODIFIED Requirements

### Requirement: LiDAR 训练集归一化与缓存
系统 MUST 支持禁用 LiDAR 全局归一化、复用 BEV 构造期固定范围输出、以及基于训练集流式估计 LiDAR BEV 通道统计量。启用训练集统计时，系统 MUST 使用内存有界的流式统计，并将同一统计量用于验证或测试 split。系统 MUST 支持将 BEV 缓存为 `.npy` 并在后续训练中按样本复用。系统 MUST NOT 在 Dataset 初始化阶段把全训练集 BEV 全量读取后通过 `concatenate`、`stack` 或等价方式一次性拼接计算 normalizer。训练流程 MUST 将 fit 后的 LiDAR normalizer/stats 保存为可复用工件；评估同一 checkpoint 时 MUST 优先复用训练时保存的 normalizer/stats。

#### Scenario: 默认禁用全局 LiDAR normalizer
- **WHEN** LiDAR 配置未显式启用训练集统计归一化
- **THEN** 系统 MUST 不扫描训练 split 来 fit LiDAR normalizer
- **AND** LiDAR BEV MUST 继续使用 BEV 构造期的固定范围、限幅或局部归一化结果

#### Scenario: 训练集流式 fit LiDAR normalizer
- **WHEN** dataloader 构建训练 split 且配置显式启用 LiDAR streaming stats 归一化
- **THEN** 系统 MUST 逐样本或逐 batch 读取训练 split 的 LiDAR BEV 并按通道累计统计量
- **AND** 系统 MUST NOT 将全部训练 BEV 同时保存在内存中
- **AND** 系统 MUST 将 fit 后的 normalizer 保存在训练 dataset 实例、可复用对象或 stats 文件中

#### Scenario: 训练保存 LiDAR normalizer 工件
- **WHEN** 启用 LiDAR streaming stats 归一化的训练流程完成 normalizer fit
- **THEN** 系统 MUST 将 fit 后的 LiDAR normalizer/stats 保存到当前运行目录的稳定工件路径，除非用户显式提供了既有 stats 文件且未要求重算
- **AND** 训练日志或 registry metadata MUST 记录该 normalizer/stats 工件路径

#### Scenario: 流式统计进度可见
- **WHEN** 系统需要在训练循环前扫描训练 split 计算 LiDAR stats 且进度显示已启用
- **THEN** 系统 MUST 提供独立进度显示或日志，说明当前处于 LiDAR stats 计算阶段

#### Scenario: 测试集复用训练 normalizer
- **WHEN** dataloader 构建测试 split 且启用 LiDAR 归一化
- **THEN** 系统 MUST 使用训练 split 已 fit、已保存或已加载的 normalizer 转换测试 LiDAR BEV
- **AND** 系统 MUST 不在测试 split 上重新 fit normalizer

#### Scenario: 评估从 checkpoint metadata 加载 normalizer
- **WHEN** 评估入口加载的 checkpoint metadata 或 registry sidecar 记录了 LiDAR normalizer/stats 路径
- **THEN** 系统 MUST 加载该 normalizer/stats 并传递给测试 dataset
- **AND** 系统 MUST 不为了 LiDAR normalizer 重新扫描训练 split

#### Scenario: 从 stats 文件复用 normalizer
- **WHEN** LiDAR 配置提供已存在的 stats 文件
- **THEN** 系统 MUST 从该文件加载 mean/std 或等价统计量
- **AND** 系统 MUST 不重新扫描训练 split，除非配置显式要求重新计算

#### Scenario: 读取 BEV 缓存
- **WHEN** LiDAR 配置启用缓存并且对应 `.npy` BEV 文件已存在
- **THEN** dataset MUST 直接读取缓存 BEV
- **AND** 读取结果 MUST 与在线构造路径保持相同 shape 和 dtype
- **AND** 缓存读取 MUST 按当前样本或当前帧发生，不得在 dataset 初始化阶段全量载入缓存目录
