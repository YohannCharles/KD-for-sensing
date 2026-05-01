## MODIFIED Requirements

### Requirement: LiDAR BEV cache 训练复用
LiDAR-only 和包含 LiDAR 的 fusion 训练配置 MUST 支持复用 BEV 磁盘缓存，以避免每个 epoch 重复执行点云到 BEV 的转换。cache 读取和写入 MUST 按样本或帧懒加载发生，不得在 dataset 初始化阶段全量读取 cache 目录或将全训练集 BEV materialize 到内存中。系统 MUST 提供可单独运行的 cache 预热入口，预热入口生成的 cache MUST 可被训练和评估配置直接复用。

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

#### Scenario: 预热 train/test LiDAR cache
- **WHEN** 用户运行 LiDAR BEV cache 预处理入口并提供 train/test CSV 或多个 CSV
- **THEN** 系统 MUST 为这些 CSV 中唯一 LiDAR 帧路径生成 BEV cache
- **AND** 已存在且参数匹配的 cache 文件 MUST 被跳过，除非用户显式启用 overwrite
- **AND** 预处理入口 MUST 显示进度或日志，并返回生成数量、跳过数量和 cache 目录

#### Scenario: LiDAR cache metadata 可追踪
- **WHEN** LiDAR BEV cache 预处理入口完成
- **THEN** 系统 MUST 在参数 hash cache 目录写出 metadata
- **AND** metadata MUST 记录 BEV size、ROI、FoV、ground/background 参数、源 CSV、生成数量、跳过数量和 cache version
- **AND** 训练运行日志或最终配置 MUST 能记录实际使用的 cache 目录和关键参数
