## ADDED Requirements

### Requirement: LiDAR BEV cache 训练复用
LiDAR-only 和包含 LiDAR 的 fusion 训练配置 MUST 支持复用 BEV 磁盘缓存，以避免每个 epoch 重复执行点云到 BEV 的转换。cache 读取和写入 MUST 按样本或帧懒加载发生，不得在 dataset 初始化阶段全量读取 cache 目录或将全训练集 BEV materialize 到内存中。

#### Scenario: LiDAR 配置启用 cache 读取
- **WHEN** 用户运行 LiDAR-only 或包含 LiDAR 的 fusion canonical 配置且配置提供 `lidar_cache_dir`
- **THEN** dataset MUST 在对应 BEV cache 文件存在时直接读取 cache
- **AND** cache 读取 MUST 只发生在当前样本被取样时
- **AND** 返回张量 shape 和 dtype MUST 与在线点云转 BEV 路径一致

#### Scenario: LiDAR 配置写入缺失 cache
- **WHEN** 用户显式启用 `lidar_write_cache` 且当前样本的 BEV cache 不存在
- **THEN** dataset MAY 在线构造该样本 BEV 并写入 cache
- **AND** 写入 MUST 不改变当前样本的 label、序列长度或返回张量契约

#### Scenario: cache 参数隔离
- **WHEN** LiDAR BEV size、ROI、FoV、ground filter 或背景过滤参数发生变化
- **THEN** 系统 MUST 避免错误复用不兼容 cache
- **AND** cache 路径、cache key 或配置约束 MUST 能区分这些 BEV 构造参数

#### Scenario: dataset 初始化不预加载 cache
- **WHEN** 用户构建启用 LiDAR cache 的训练 dataset
- **THEN** dataset 初始化 MUST 不遍历 cache 目录读取所有 `.npy` 文件
- **AND** dataset 初始化 MUST 不创建包含全训练集 BEV 的 list、ndarray 或 tensor

### Requirement: LiDAR cache 预处理入口兼容训练配置
LiDAR BEV cache 预处理入口生成的 cache MUST 可被训练和评估配置直接复用。训练配置中的 cache 目录、BEV size、ROI、FoV 和过滤参数 MUST 与预处理配置保持可追踪一致。

#### Scenario: 预处理生成后训练复用
- **WHEN** 用户先运行 LiDAR BEV cache 预处理入口生成 cache，再运行 LiDAR-only 训练
- **THEN** 训练 dataset MUST 能读取预处理生成的 cache
- **AND** 训练运行日志或最终配置 MUST 记录 cache 目录和关键 BEV 参数

#### Scenario: cache 缺失回退在线构造
- **WHEN** 配置启用 `lidar_use_cache` 但某个样本的 cache 文件不存在
- **THEN** 如果 `lidar_write_cache` 为 true，dataset MUST 在线构造并写入该样本 cache
- **AND** 如果 `lidar_write_cache` 为 false，dataset MUST 在线构造该样本 BEV 或抛出包含 cache 路径的清晰错误，具体行为 MUST 由配置显式决定
