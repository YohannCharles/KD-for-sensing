# lidar-preprocessing Specification

## Purpose
定义 LiDAR 点云读取、BEV cache、normalization artifact 和序列列生成要求。
## Requirements
### Requirement: LiDAR 序列列生成
系统 MUST 在 Scenario 9 序列生成流程中支持保留 LiDAR 路径列，使启用 LiDAR 的 dataset 能按历史时隙读取点云或 BEV 缓存。序列 CSV MUST 至少包含每个历史时隙对应的 `lidar1..lidarN` 列，并保持既有 camera、radar、GPS、beam、future_beam 和 `seq_index` 列兼容。

#### Scenario: 生成携带 LiDAR 的序列 CSV
- **WHEN** 用户通过预处理入口从原始 scenario CSV 生成输入长度为 8、预测长度为 3 的序列 CSV，并启用 LiDAR 列输出
- **THEN** 系统 MUST 生成 `lidar1` 到 `lidar8` 的 LiDAR 路径列
- **AND** 系统 MUST 保留既有 camera、radar、beam、future_beam 和 `seq_index` 列

#### Scenario: 旧序列 CSV 兼容非 LiDAR 实验
- **WHEN** 用户使用不包含 LiDAR 列的旧序列 CSV 运行 image-only、radar-only、GPS-only 或未启用 LiDAR 的 fusion 配置
- **THEN** 系统 MUST 不要求 CSV 中存在 LiDAR 列
- **AND** 系统 MUST 保持既有数据加载行为不变

### Requirement: LiDAR 点云读取与过滤
系统 MUST 能从 LiDAR 路径读取点云数据，并在构造特征前过滤 NaN、inf 和无效坐标。读取失败、缺失文件或格式非法时，启用 LiDAR 的配置 MUST 抛出包含路径信息的清晰错误。

#### Scenario: 读取 ASCII 点云
- **WHEN** dataset 读取一个包含 `x y z intensity` 或 `x y z` 数值列的 ASCII PCD、文本、CSV 或 `.npy` 点云文件
- **THEN** 系统 MUST 返回形状为 `[N, 4]` 的浮点数组
- **AND** 缺失 intensity 时系统 MUST 使用默认强度值补齐第四列

#### Scenario: 过滤无效点
- **WHEN** 点云包含 NaN、inf 或坐标缺失的点
- **THEN** 系统 MUST 在 BEV 构造前移除这些点
- **AND** 有效点为空时系统 MUST 返回全零 BEV 张量而不是中断训练

#### Scenario: LiDAR 文件缺失
- **WHEN** 启用 LiDAR 的 dataset 遇到不存在的 LiDAR 路径
- **THEN** 系统 MUST 抛出包含缺失路径的异常

### Requirement: LiDAR BEV 伪图像构造
系统 MUST 支持将 LiDAR 点云转换为固定尺寸 BEV 伪图像。默认 BEV MUST 使用三个通道表示 height、intensity 和 density，并输出可被 CNN 模型消费的 `[channels, height, width]` 浮点数组。

#### Scenario: 构造三通道 BEV
- **WHEN** 点云经过 ROI/FoV 裁剪后落入 BEV 网格
- **THEN** 系统 MUST 输出形状为 `[3, bev_height, bev_width]` 的 `float32` 数组
- **AND** channel 1 MUST 表示归一化高度
- **AND** channel 2 MUST 表示归一化强度
- **AND** channel 3 MUST 表示 log-normalized 点密度

#### Scenario: ROI 和 FoV 裁剪
- **WHEN** LiDAR 配置提供 ROI 或 FoV 范围
- **THEN** 系统 MUST 仅使用范围内点构造 BEV
- **AND** 输出 BEV 尺寸 MUST 不随点数量或裁剪结果变化

#### Scenario: 空裁剪结果
- **WHEN** ROI/FoV 裁剪后没有点保留
- **THEN** 系统 MUST 返回固定尺寸全零 BEV
- **AND** 系统 MUST 不改变该样本的 label 或序列长度

### Requirement: LiDAR 背景过滤与安全增强
系统 MUST 将背景过滤和 LiDAR 数据增强作为显式配置能力。背景过滤 MUST 是可选项；默认训练增强 MUST 只包含不会改变 beam label 的点 dropout/downsampling 和小幅 3D jitter。

#### Scenario: 禁用背景过滤
- **WHEN** LiDAR 配置未启用背景过滤
- **THEN** 系统 MUST 仅执行无效点过滤、ROI/FoV 裁剪和 BEV 构造

#### Scenario: 启用背景过滤
- **WHEN** LiDAR 配置启用背景过滤并提供场景级背景统计或缓存
- **THEN** 系统 MUST 在 BEV 构造前移除稳定背景点
- **AND** 系统 MUST 保留不满足背景稳定条件的潜在车辆或遮挡物点

#### Scenario: 训练增强不改 label
- **WHEN** dataset split 为 train 且启用 LiDAR 增强
- **THEN** 系统 MUST 只应用点 dropout/downsampling 和小幅位置 jitter
- **AND** 系统 MUST 不执行需要同步修改 beam label 的水平翻转增强

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

### Requirement: LiDAR batch 字段
启用 LiDAR 的 Scenario 9 dataset MUST 在样本字典中返回 `lidar` 字段。`lidar` 字段 MUST 为浮点张量，形状为 `[seq_len, channels, height, width]`，并与 `input_beam` 的历史时隙对齐。

#### Scenario: dataset 返回 LiDAR 张量
- **WHEN** dataset 配置启用 LiDAR 且读取一个序列样本
- **THEN** 返回样本 MUST 包含 `lidar`
- **AND** `lidar` MUST 是 `torch.float32` 张量
- **AND** `lidar` 的第一维长度 MUST 等于配置的 `seq_len`
- **AND** `lidar` 的通道维长度 MUST 等于 LiDAR BEV 通道数

#### Scenario: dataset 不启用 LiDAR
- **WHEN** dataset 配置未启用 LiDAR
- **THEN** 返回样本 MUST 不要求包含 `lidar`
- **AND** 训练、验证和评估流程 MUST 保持旧配置兼容

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

### Requirement: LiDAR BEV cache 原子写入
LiDAR BEV cache 写入 MUST 避免其它并行训练或评估进程读取到半成品文件。系统 MUST 使用临时文件加原子替换、文件锁或等价机制写入 `.npy` cache。

#### Scenario: 并发写入同一 LiDAR cache
- **WHEN** 两个训练进程在 `auto` policy 下同时遇到同一个缺失 LiDAR BEV cache
- **THEN** 任一进程写入时 MUST 不暴露半写入目标文件
- **AND** 最终存在的 cache 文件 MUST 可被 `np.load` 正常读取

#### Scenario: 读取已完成 LiDAR cache
- **WHEN** LiDAR BEV cache 文件已经完成写入
- **THEN** 后续 dataset 取样 MUST 直接读取该 cache
- **AND** 读取结果 MUST 保持与在线构造路径相同的 shape 和 dtype

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

### Requirement: LiDAR cache 与 ROI/FoV 参数可追踪
LiDAR baseline profile MUST 使用参数隔离的 BEV cache，并 MUST 在训练输出中记录实际使用的 ROI、FoV、ground/background filter 和 cache 目录。

#### Scenario: cache key 区分 LiDAR 构造参数
- **WHEN** LiDAR BEV size、ROI、FoV、ground filter 或 background filter 参数变化
- **THEN** 系统 MUST 使用不同 cache key、目录或 metadata 约束避免错误复用旧 BEV cache

#### Scenario: final_config 记录 LiDAR profile
- **WHEN** 一次启用 LiDAR 的训练启动
- **THEN** final_config 或运行 metadata MUST 记录 LiDAR BEV size、ROI、FoV、normalization、cache policy 和 cache 目录

### Requirement: LiDAR 预处理示例不依赖 KD 命名
LiDAR preprocessing、cache 和 normalization 文档或示例配置 MUST 使用 supervised/strong/lightweight 命名引用训练入口，不得推荐 `*_no_kd`、`logits_kd` 或 `rkd` 路径。

#### Scenario: LiDAR preprocessing 后续训练提示
- **WHEN** 文档或 CLI 输出提示用户运行 LiDAR 训练
- **THEN** 推荐路径 MUST 使用 LiDAR supervised、strong 或 lightweight 配置
- **AND** 推荐路径 MUST 不包含 KD token

### Requirement: LiDAR BEV grid metadata for current consumers
LiDAR BEV cache MUST 记录足以复现当前 LiDAR baseline、diagnostic 和 future attention consumer 的 BEV metadata。metadata MUST 包含 ROI、BEV size、grid size、cell center convention、FoV、ground/background filter、cache version 和参数 hash。旧 BGAM angle-grid consumer 已退役，MUST NOT 作为当前 metadata 需求来源。

#### Scenario: 当前 LiDAR consumer 使用 BEV cache metadata
- **WHEN** 当前 LiDAR baseline、diagnostic 或 attention prototype 读取 LiDAR BEV cache
- **THEN** 系统 MUST 能获得该 BEV 的 ROI、height、width 和 cell center convention
- **AND** run metadata MUST 记录实际使用的 BEV profile 和 cache hash
- **AND** 实现 MUST NOT 依赖旧 `GPSGuidedBGAM` 或 `gps_lidar_bgam` module path

#### Scenario: cache 参数不匹配
- **WHEN** 当前 manifest、配置或 diagnostic 所需 BEV grid 参数与 cache metadata 不一致
- **THEN** 系统 MUST 拒绝复用该 cache 或按配置在线重建
- **AND** 错误信息 MUST 包含不匹配的参数名

### Requirement: 未接入 LiDAR pillar encoder 原型不属于当前支持面
当前 LiDAR preprocessing support surface MUST 以点云读取、BEV 伪图像构造、cache、normalization、质量摘要和启用 LiDAR 的 dataset flat sample 为准。未注册、未配置、未被训练/评估/诊断入口消费的 pillar encoder 或 spatial encoder 原型 MUST 不作为当前必须保留的 LiDAR 能力。

#### Scenario: 删除未接入 pillar encoder
- **WHEN** `lidar_pillar_encoder` 或等价原型没有 registry、config、dataset、trainer、CLI、README/docs 或 current OpenSpec 消费
- **THEN** 本 change MAY 删除该源码模块
- **AND** LiDAR BEV 构造、cache 预热、normalization 和质量摘要 MUST 保持可用

#### Scenario: 新增 pillar 能力必须重新走 OpenSpec
- **WHEN** 后续需要 `lidar.profile=pillar6`、pillar scatter 或 LiDAR spatial encoder 作为当前训练能力
- **THEN** 项目 MUST 通过新的 OpenSpec change 声明模型注册、配置入口、dataset contract、forward metadata 和 focused tests
- **AND** 不得通过恢复未接入原型文件把该能力静默加入 current surface

### Requirement: LiDAR debug quality summary
当前 LiDAR baseline、diagnostic 或 attention prototype MUST 能记录 LiDAR BEV 输入质量摘要，以便判断模型或诊断是否被空 BEV、极端稀疏或 cache 混用影响。旧 BGAM debug mask/report 已退役，MUST NOT 作为当前质量摘要入口。

#### Scenario: 记录 LiDAR 质量
- **WHEN** 当前 LiDAR consumer 完成训练、评估或诊断
- **THEN** run metadata 或 diagnostics MUST 记录 raw/model input BEV 非空率、通道均值/标准差、零值比例、ROI、BEV size 和 cache path
- **AND** 质量摘要 MUST 区分 raw BEV 与模型实际输入 feature

#### Scenario: 标记退化风险
- **WHEN** LiDAR 质量摘要显示大量全零帧、近常量通道或 cache 参数异常
- **THEN** 系统 MUST 在 run metadata 或 report 中标记 `lidar_input_degradation_risk`
- **AND** report MUST 给出相关参数和受影响样本数
