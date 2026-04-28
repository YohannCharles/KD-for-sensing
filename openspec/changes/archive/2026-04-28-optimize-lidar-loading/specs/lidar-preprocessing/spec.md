## ADDED Requirements

### Requirement: LiDAR Dataset 懒加载初始化
启用 LiDAR 的 Scenario 9 dataset MUST 在初始化阶段保持轻量。初始化阶段 MUST 只解析 CSV、保存路径、校验必要列和加载小型配置或 stats 文件；不得遍历全部样本读取 LiDAR 点云、构造 BEV、读取全部 BEV cache，或把训练集 LiDAR BEV materialize 到内存中。

#### Scenario: 初始化不读取全量 LiDAR
- **WHEN** 用户构建启用 LiDAR 的训练 dataset
- **THEN** dataset 初始化 MUST 不调用逐样本 LiDAR BEV 构造逻辑遍历整个训练 split
- **AND** dataset 初始化 MUST 不创建包含全训练集 LiDAR BEV 的 list、ndarray 或 tensor

#### Scenario: 第一次取样时读取 LiDAR
- **WHEN** 用户从启用 LiDAR 的 dataset 读取单个样本
- **THEN** 系统 MUST 仅为该样本历史时隙读取点云或 BEV cache
- **AND** 返回的 `lidar` 张量 MUST 与既有 `[seq_len, channels, height, width]` 契约一致

#### Scenario: 初始化可加载小型 stats 文件
- **WHEN** LiDAR 配置指定已存在的归一化 stats 文件
- **THEN** dataset 初始化 MAY 加载该小型 stats 文件
- **AND** dataset 初始化 MUST 不因此读取任何样本级 LiDAR 点云或 BEV 数据

## MODIFIED Requirements

### Requirement: LiDAR 训练集归一化与缓存
系统 MUST 支持禁用 LiDAR 全局归一化、复用 BEV 构造期固定范围输出、以及基于训练集流式估计 LiDAR BEV 通道统计量。启用训练集统计时，系统 MUST 使用内存有界的流式统计，并将同一统计量用于验证或测试 split。系统 MUST 支持将 BEV 缓存为 `.npy` 并在后续训练中按样本复用。系统 MUST NOT 在 Dataset 初始化阶段把全训练集 BEV 全量读取后通过 `concatenate`、`stack` 或等价方式一次性拼接计算 normalizer。

#### Scenario: 默认禁用全局 LiDAR normalizer
- **WHEN** LiDAR 配置未显式启用训练集统计归一化
- **THEN** 系统 MUST 不扫描训练 split 来 fit LiDAR normalizer
- **AND** LiDAR BEV MUST 继续使用 BEV 构造期的固定范围、限幅或局部归一化结果

#### Scenario: 训练集流式 fit LiDAR normalizer
- **WHEN** dataloader 构建训练 split 且配置显式启用 LiDAR streaming stats 归一化
- **THEN** 系统 MUST 逐样本或逐 batch 读取训练 split 的 LiDAR BEV 并按通道累计统计量
- **AND** 系统 MUST NOT 将全部训练 BEV 同时保存在内存中
- **AND** 系统 MUST 将 fit 后的 normalizer 保存在训练 dataset 实例、可复用对象或 stats 文件中

#### Scenario: 流式统计进度可见
- **WHEN** 系统需要在训练循环前扫描训练 split 计算 LiDAR stats 且进度显示已启用
- **THEN** 系统 MUST 提供独立进度显示或日志，说明当前处于 LiDAR stats 计算阶段

#### Scenario: 测试集复用训练 normalizer
- **WHEN** dataloader 构建测试 split 且启用 LiDAR 归一化
- **THEN** 系统 MUST 使用训练 split 已 fit 或已加载的 normalizer 转换测试 LiDAR BEV
- **AND** 系统 MUST 不在测试 split 上重新 fit normalizer

#### Scenario: 从 stats 文件复用 normalizer
- **WHEN** LiDAR 配置提供已存在的 stats 文件
- **THEN** 系统 MUST 从该文件加载 mean/std 或等价统计量
- **AND** 系统 MUST 不重新扫描训练 split，除非配置显式要求重新计算

#### Scenario: 读取 BEV 缓存
- **WHEN** LiDAR 配置启用缓存并且对应 `.npy` BEV 文件已存在
- **THEN** dataset MUST 直接读取缓存 BEV
- **AND** 读取结果 MUST 与在线构造路径保持相同 shape 和 dtype
- **AND** 缓存读取 MUST 按当前样本或当前帧发生，不得在 dataset 初始化阶段全量载入缓存目录
