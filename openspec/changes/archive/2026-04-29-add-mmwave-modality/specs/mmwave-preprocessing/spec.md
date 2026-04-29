## ADDED Requirements

### Requirement: mmWave 序列列生成
系统 MUST 在 Scenario 9 序列生成流程中支持保留 mmWave receive-power 路径列，使启用 mmWave 的 dataset 能按历史时隙读取 64 维 power vector。序列 CSV MUST 至少包含每个历史时隙对应的 `mmwave1..mmwaveN` 列，并保持既有 camera、radar、GPS、LiDAR、beam、future_beam 和 `seq_index` 列兼容。

#### Scenario: 生成携带 mmWave 的序列 CSV
- **WHEN** 用户通过预处理入口从原始 scenario CSV 生成输入长度为 8、预测长度为 3 的序列 CSV，并启用 mmWave 列输出
- **THEN** 系统 MUST 生成 `mmwave1` 到 `mmwave8` 的 mmWave power 路径列
- **AND** 默认 mmWave 路径来源 MUST 为 `unit1_pwr_60ghz`
- **AND** 系统 MUST 保留既有 camera、radar、GPS、LiDAR、beam、future_beam 和 `seq_index` 列

#### Scenario: 旧序列 CSV 兼容非 mmWave 实验
- **WHEN** 用户使用不包含 mmWave 列的旧序列 CSV 运行 image-only、radar-only、GPS-only、LiDAR-only 或未启用 mmWave 的 fusion 配置
- **THEN** 系统 MUST 不要求 CSV 中存在 mmWave 列
- **AND** 系统 MUST 保持既有数据加载行为不变

### Requirement: mmWave power vector 读取与清洗
系统 MUST 能从 mmWave 路径读取 DeepSense 60GHz receive-power vector，并将其转换为固定 64 维浮点数组。读取失败、缺失文件、维度非法或格式非法时，启用 mmWave 的配置 MUST 抛出包含路径信息的清晰错误。

#### Scenario: 读取 64 维 power 文本
- **WHEN** dataset 读取一个包含 64 个数值的 mmWave power txt 文件
- **THEN** 系统 MUST 返回形状为 `[64]` 的 `float32` 数组
- **AND** 系统 MUST 支持每行一个数值或等价可由 NumPy 解析的数值文本格式

#### Scenario: 清洗无效 power 值
- **WHEN** power vector 包含 NaN、inf 或非正值
- **THEN** 系统 MUST 在 dB 压缩前进行 finite 清洗和 epsilon 裁剪
- **AND** 输出特征 MUST 不包含 NaN 或 inf

#### Scenario: mmWave 文件缺失
- **WHEN** 启用 mmWave 的 dataset 遇到不存在的 mmWave 路径
- **THEN** 系统 MUST 抛出包含缺失路径的异常

#### Scenario: mmWave 维度非法
- **WHEN** mmWave power 文件解析后不是 64 个数值
- **THEN** 系统 MUST 拒绝该样本
- **AND** 错误信息 MUST 包含实际数值个数和期望维度 64

### Requirement: mmWave dB 特征与训练集 scaler
系统 MUST 将 mmWave receive-power vector 转换为 dB 压缩后的 64 维特征，并支持只使用训练 split fit 的 z-score scaler。系统 MUST 不使用验证集或测试集统计量参与 mmWave 特征归一化。训练流程 MUST 将 fit 后的 mmWave scaler 作为可复用工件保存，并在评估同一 checkpoint 时优先加载训练时保存的 scaler。

#### Scenario: 构造 mmWave dB 特征
- **WHEN** dataset 加载一个 mmWave power vector
- **THEN** 系统 MUST 输出 64 维 dB 压缩特征
- **AND** dB 压缩 MUST 在 finite 清洗和 epsilon 裁剪之后执行

#### Scenario: 训练集 fit mmWave scaler
- **WHEN** dataloader 构建训练 split 且启用 mmWave 归一化
- **THEN** 系统 MUST 使用训练 split 的历史 mmWave 特征 fit scaler
- **AND** 系统 MUST 不使用 `future_beam` 对应的未来 power vector fit 输入 scaler

#### Scenario: 训练保存 mmWave scaler 工件
- **WHEN** 启用 mmWave 归一化的训练流程完成 dataloader 构建
- **THEN** 系统 MUST 将训练集 fit 后的 mmWave scaler 保存到当前运行目录的稳定工件路径
- **AND** 训练日志或 registry metadata MUST 记录该 scaler 工件路径

#### Scenario: 测试集复用训练 mmWave scaler
- **WHEN** dataloader 构建测试 split 且启用 mmWave 归一化
- **THEN** 系统 MUST 使用训练 split 已 fit 或从训练工件加载的 scaler 转换测试 mmWave 特征
- **AND** 系统 MUST 不在测试 split 上重新 fit scaler

#### Scenario: 缺少 scaler 的 mmWave 评估
- **WHEN** 评估入口直接构建启用 mmWave 归一化的测试 dataset 且没有可用训练 scaler、scaler 文件或 registry metadata
- **THEN** 系统 MUST 抛出清晰错误，提示需要提供训练集 mmWave scaler 或使用训练 dataloader 构建流程

### Requirement: mmWave batch 字段
启用 mmWave 的 Scenario 9 dataset MUST 在样本字典中返回 `mmwave` 字段。`mmwave` 字段 MUST 为浮点张量，形状为 `[seq_len, 64]`，并与 `input_beam` 的历史时隙对齐。

#### Scenario: dataset 返回 mmWave 张量
- **WHEN** dataset 配置启用 mmWave 且读取一个序列样本
- **THEN** 返回样本 MUST 包含 `mmwave`
- **AND** `mmwave` MUST 是 `torch.float32` 张量
- **AND** `mmwave` 的第一维长度 MUST 等于配置的 `seq_len`
- **AND** `mmwave` 的第二维长度 MUST 为 64

#### Scenario: dataset 不启用 mmWave
- **WHEN** dataset 配置未启用 mmWave
- **THEN** 返回样本 MUST 不要求包含 `mmwave`
- **AND** 训练、验证和评估流程 MUST 保持旧配置兼容
