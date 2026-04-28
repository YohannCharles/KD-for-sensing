## ADDED Requirements

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

### Requirement: LiDAR 训练集归一化与缓存
系统 MUST 支持基于训练集估计 LiDAR BEV 通道统计量，并将同一统计量用于验证或测试 split。系统 MUST 支持将 BEV 缓存为 `.npy` 并在后续训练中复用。

#### Scenario: 训练集 fit LiDAR normalizer
- **WHEN** dataloader 构建训练 split 且启用 LiDAR 归一化
- **THEN** 系统 MUST 使用训练 split 的 LiDAR BEV 统计量 fit normalizer
- **AND** 系统 MUST 将 fit 后的 normalizer 保存在训练 dataset 实例或可复用对象中

#### Scenario: 测试集复用训练 normalizer
- **WHEN** dataloader 构建测试 split 且启用 LiDAR 归一化
- **THEN** 系统 MUST 使用训练 split 已 fit 的 normalizer 转换测试 LiDAR BEV
- **AND** 系统 MUST 不在测试 split 上重新 fit normalizer

#### Scenario: 读取 BEV 缓存
- **WHEN** LiDAR 配置启用缓存并且对应 `.npy` BEV 文件已存在
- **THEN** dataset MUST 直接读取缓存 BEV
- **AND** 读取结果 MUST 与在线构造路径保持相同 shape 和 dtype

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
