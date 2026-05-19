## ADDED Requirements

### Requirement: Snapshot next-frame 序列预处理
序列 CSV 预处理 MUST 支持 snapshot next-frame 协议。该协议 MUST 在 Scenario 31 上使用 `in_len=1` 和 `out_len=1` 生成所有合法窗口，并保留当前帧输入与下一帧监督目标所需的列。

#### Scenario: 生成 snapshot 窗口
- **WHEN** 用户运行 snapshot next-frame 预处理配置
- **THEN** 预处理 MUST 设置 `in_len: 1` 和 `out_len: 1`
- **AND** 每个窗口 MUST 包含 `camera1`、`radar1`、`gps1`、`bs_gps1`、`lidar1`、`mmwave1`、`beam1`、`future_beam1` 和 `seq_index`
- **AND** 窗口 MUST 不跨 `seq_index` 拼接

#### Scenario: 生成 position target 列
- **WHEN** snapshot 预处理配置启用 `include_position_targets: true`
- **THEN** 输出 CSV MUST 包含 `future_gps1` 和 `future_bs_gps1`
- **AND** position objective MUST 使用这两个 future 列构造下一帧位置 target

### Requirement: Snapshot 80/20 sequence split
Snapshot next-frame 预处理 MUST 以完整 `seq_index` 为单位生成 80% train 和 20% validation split。split metadata MUST 明确记录这是 snapshot 协议，不得伪装成历史窗口统一 split。

#### Scenario: 写出 train/validation CSV
- **WHEN** snapshot next-frame 预处理完成
- **THEN** 系统 MUST 写出 `train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 系统 MUST 写出 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** validation 窗口数量 MUST 尽量接近全量 snapshot 窗口的 20%

#### Scenario: split metadata 记录 snapshot 协议
- **WHEN** snapshot next-frame 预处理写出 split metadata
- **THEN** metadata MUST 包含 `split_protocol: snapshot_next_frame_balanced_seq`
- **AND** metadata MUST 包含 `training_set_pct: 0.8`
- **AND** metadata MUST 包含 train/validation `seq_index` 列表、窗口数、输出 CSV 路径和 label 分布摘要

#### Scenario: 不复用历史窗口 metadata
- **WHEN** snapshot 配置构建 dataset
- **THEN** 数据构建流程 MUST 加载 snapshot split metadata
- **AND** 如果 metadata 显示 `in_len` 或 `out_len` 不是 1，系统 MUST 拒绝该配置

### Requirement: Snapshot split-dependent artifact 隔离
依赖训练 split 拟合的 normalization、cache stats 或 target stats MUST 基于 snapshot train split 重新 fit。评估或验证 snapshot checkpoint 时，系统 MUST 使用同一 snapshot run 保存的 artifact 或与 snapshot split fingerprint 匹配的 artifact。

#### Scenario: mmWave scaler 使用 snapshot train split
- **WHEN** snapshot 配置启用 mmWave normalization
- **THEN** mmWave scaler MUST 只使用 `train_seqs_SNAPSHOT_NEXT_FRAME.csv` 中的 `mmwave1` fit
- **AND** validation dataset MUST 复用训练 split scaler
- **AND** 系统 MUST 不复用历史窗口 run 的 mmWave scaler

#### Scenario: LiDAR streaming stats 使用 snapshot train split
- **WHEN** snapshot 配置启用 LiDAR streaming stats
- **THEN** LiDAR normalizer MUST 只基于 snapshot train split 拟合统计量
- **AND** artifact metadata MUST 记录 snapshot split metadata path 和 fingerprint

#### Scenario: target stats 使用 snapshot train split
- **WHEN** snapshot objective 需要 occlusion threshold 或 position target scaler
- **THEN** threshold 或 scaler MUST 只从 snapshot train split 拟合
- **AND** validation split MUST 不参与拟合

#### Scenario: frame-level cache 可复用
- **WHEN** LiDAR BEV cache 或等价帧级 cache 只依赖原始文件路径和预处理参数
- **THEN** snapshot 配置 MAY 复用既有帧级 cache
- **AND** split-dependent normalizer/stat artifact MUST 仍与 snapshot split fingerprint 绑定
