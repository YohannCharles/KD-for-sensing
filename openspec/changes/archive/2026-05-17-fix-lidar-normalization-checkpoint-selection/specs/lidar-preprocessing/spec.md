## MODIFIED Requirements

### Requirement: LiDAR 训练集归一化与缓存
系统 MUST 支持禁用 LiDAR 全局归一化、复用 BEV 构造期固定范围输出、以及基于训练集流式估计 LiDAR BEV 通道统计量。启用训练集统计时，系统 MUST 使用内存有界的流式统计，并将同一统计量用于验证或测试 split。系统 MUST 支持将 BEV 缓存为 `.npy` 并在后续训练中按样本复用。系统 MUST NOT 在 Dataset 初始化阶段把全训练集 BEV 全量读取后通过 `concatenate`、`stack` 或等价方式一次性拼接计算 normalizer。训练流程 MUST 将 fit 后的 LiDAR normalizer/stats 保存为可复用工件；评估同一 checkpoint 时 MUST 优先复用训练时保存的 normalizer/stats。`lidar_normalize` 与 `lidar_normalization.enabled` MUST 在最终配置中表达同一状态；系统 MUST NOT 在两者冲突时静默启用 streaming stats。

#### Scenario: 默认禁用全局 LiDAR normalizer
- **WHEN** LiDAR 配置未显式启用训练集统计归一化
- **THEN** 系统 MUST 不扫描训练 split 来 fit LiDAR normalizer
- **AND** LiDAR BEV MUST 继续使用 BEV 构造期的固定范围、限幅或局部归一化结果
- **AND** 最终配置、dataset metadata 和诊断报告 MUST 将 normalization profile 记录为 `bev_raw` 或等价 raw 标识

#### Scenario: 拒绝冲突的 LiDAR normalization 配置
- **WHEN** 配置同时提供 `lidar_normalize` 和 `lidar_normalization.enabled` 且两者布尔值不一致
- **THEN** 配置解析或 dataset 构造 MUST 拒绝该配置
- **AND** 错误信息 MUST 同时指出两个字段的值
- **AND** 错误信息 MUST 提示用户选择 raw BEV 或显式 streaming stats profile

#### Scenario: 训练集流式 fit LiDAR normalizer
- **WHEN** dataloader 构建训练 split 且配置显式启用 LiDAR streaming stats 归一化
- **THEN** 系统 MUST 逐样本或逐 batch 读取训练 split 的 LiDAR BEV 并按通道累计统计量
- **AND** 系统 MUST NOT 将全部训练 BEV 同时保存在内存中
- **AND** 系统 MUST 将 fit 后的 normalizer 保存在训练 dataset 实例、可复用对象或 stats 文件中
- **AND** 最终配置、dataset metadata 和诊断报告 MUST 将 normalization profile 记录为 `bev_streaming_stats` 或等价 stats 标识

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

### Requirement: LiDAR baseline profile 显式启用训练集统计归一化
LiDAR-only teacher baseline 和包含 LiDAR 的 fusion baseline 配置 MUST 使用内部一致的 normalization 配置。默认 baseline MUST 使用 raw BEV，除非配置文件、profile 名称或命令行覆盖显式启用训练集 streaming stats 归一化。显式启用 streaming stats 时，系统 MUST 将同一 normalizer 复用于测试 split，并 MUST 在最终配置中避免 `lidar_normalize` 与 `lidar_normalization.enabled` 冲突。

#### Scenario: 默认 LiDAR teacher baseline 使用 raw BEV
- **WHEN** 用户运行默认 LiDAR teacher/no-KD baseline 配置且未显式启用 LiDAR streaming stats 归一化
- **THEN** 配置 MUST 将 LiDAR normalization 解析为 disabled
- **AND** dataset MUST 不 fit LiDAR streaming stats normalizer
- **AND** `final_config.yaml` MUST 不出现 `lidar_normalize: false` 与 `lidar_normalization.enabled: true` 的冲突组合

#### Scenario: 显式 LiDAR stats profile 启用 streaming stats
- **WHEN** 用户运行显式启用 LiDAR stats 的 baseline、ablation profile 或命令行覆盖
- **THEN** 配置 MUST 将 LiDAR normalization 解析为 enabled
- **AND** normalization mode MUST 为 `streaming_stats` 或等价训练集统计模式
- **AND** 系统 MUST 只在 train split 上 fit normalizer
- **AND** test split MUST 复用 train split fit 得到的 normalizer

#### Scenario: LiDAR stats 工件可复用
- **WHEN** LiDAR stats profile 训练完成 normalizer fit
- **THEN** 系统 MUST 将 LiDAR stats 或 normalizer 保存为运行工件
- **AND** 评估同一 checkpoint 时 MUST 优先复用该工件
- **AND** 系统 MUST 不在 test split 上重新 fit LiDAR normalizer

### Requirement: LiDAR baseline 输入质量诊断
LiDAR baseline 训练和评估 MUST 记录 BEV 输入质量摘要，使空 BEV、近常量通道、cache 参数混用、ROI/FoV 异常或 normalization 后异常幅值可以被定位。质量摘要 MUST 区分 raw BEV 统计与模型实际输入统计；当模型输入经过 streaming stats 归一化时，raw BEV 的稀疏度 MUST 仍可见。

#### Scenario: 记录 raw BEV 非空率和通道统计
- **WHEN** 用户运行启用 LiDAR 的训练或评估
- **THEN** 系统 MUST 记录 raw LiDAR BEV 非空帧比例
- **AND** 系统 MUST 记录 raw BEV 每个通道的均值、标准差和零值比例摘要
- **AND** 摘要 MUST 区分 train/test split 或在 metadata 中标明来源 split

#### Scenario: 记录模型输入统计
- **WHEN** 用户运行启用 LiDAR 的训练或评估
- **THEN** 系统 MUST 记录模型实际接收的 LiDAR 张量统计
- **AND** 如果启用 streaming stats 归一化，模型输入统计 MUST 与 raw BEV 统计分开记录
- **AND** 归一化后 `zero_ratio` 变为非零值时，系统 MUST 仍保留 raw BEV 的 `zero_ratio`

#### Scenario: 标记疑似退化输入
- **WHEN** LiDAR BEV 质量摘要显示大量全零帧、raw BEV 极端稀疏、通道标准差低于实现定义的退化阈值，或归一化后幅值超过实现定义的异常阈值
- **THEN** 系统 MUST 在运行报告中标记 LiDAR input degradation risk
- **AND** 报告 MUST 包含对应的 ROI、FoV、normalization 和 cache 参数
