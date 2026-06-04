# snapshot-next-frame-baselines Specification

## Purpose
定义 snapshot next-frame baseline 的 split、配置、模型和运行产物契约。
## Requirements
### Requirement: Snapshot next-frame baseline 契约
系统 MUST 提供 snapshot next-frame baseline 实验族。该实验族 MUST 只使用当前帧输入预测下一帧目标，配置层必须设置输入历史长度为 1、预测 horizon 为 1，并且不得消费更早历史帧。

#### Scenario: 配置使用当前帧预测下一帧
- **WHEN** 用户加载任一 snapshot next-frame baseline 配置
- **THEN** 最终配置 MUST 设置 `data.dataset.seq_len: 1`
- **AND** 最终配置 MUST 设置 `data.dataset.num_pred: 1`
- **AND** 最终配置 MUST 设置 `model.seq_length: 1`
- **AND** 最终配置 MUST 设置 `model.num_pred: 1`

#### Scenario: 标签保持 future-only
- **WHEN** snapshot baseline 构建 batch labels
- **THEN** labels MUST 来自 `target_beam[:, :1]` 或当前 objective 对应的第一个 future target
- **AND** labels MUST 不包含 `input_beam` 的当前帧或任何历史 beam label

### Requirement: Snapshot 专用数据口径
Snapshot next-frame baseline MUST 默认使用由 `in_len=1/out_len=1` 预处理生成的专用 Scenario 31 CSV。系统 MUST 不默认复用历史窗口 `seq_len=8/out_len=3` CSV 作为 paper-aligned snapshot 数据口径。

#### Scenario: 使用 snapshot 专用 CSV
- **WHEN** 用户加载任一 snapshot next-frame baseline 配置
- **THEN** 最终配置 MUST 默认引用 `train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 最终配置 MUST 默认引用 `val_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 运行 metadata MUST 记录 snapshot split metadata 路径

#### Scenario: 拒绝静默回退历史窗口 CSV
- **WHEN** snapshot baseline 配置找不到 snapshot 专用 CSV
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 提示先运行 snapshot 专用预处理
- **AND** 系统 MUST 不静默回退到 `train_seqs_RA_GPS_LIDAR.csv` 或 `test_seqs_RA_GPS_LIDAR.csv`

### Requirement: Snapshot baseline 不使用时序模型
Snapshot next-frame baseline 模型 MUST 不包含 GRU、RNN、LSTM、TCN 或跨时间 self-attention。多模态 fusion 只允许在同一当前帧的不同模态 token 之间交互。

#### Scenario: 单模态模型无 GRU
- **WHEN** 用户构建单模态 snapshot baseline 模型
- **THEN** 模型 representation core MUST 为无时序 core
- **AND** 模型模块树中 MUST 不包含 `torch.nn.GRU`、`torch.nn.RNN` 或 `torch.nn.LSTM`

#### Scenario: 多模态模型无跨时间建模
- **WHEN** 用户构建多模态 snapshot fusion baseline 模型
- **THEN** fusion core MUST 只接收当前帧的模态 token
- **AND** fusion core MUST 不添加或消费多个时间步的 time embedding
- **AND** forward MUST 拒绝时间维大于 1 的输入

### Requirement: 单模态 snapshot baseline 配置矩阵
项目 MUST 为 `image`、`radar`、`gps`、`lidar` 和 `mmwave` 提供可加载的 snapshot next-frame supervised 单模态配置入口。每个入口 MUST 复用对应模态的现有 loader、encoder、normalization、loss 和指标契约。

#### Scenario: 加载 image snapshot 配置
- **WHEN** 用户加载 `configs/image/snapshot_next_frame_supervised.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: image`
- **AND** 最终配置 MUST 使用 image-only 当前帧模型
- **AND** 最终配置 MUST 不包含 `distillation.type`

#### Scenario: 加载所有单模态 snapshot 配置
- **WHEN** 开发者加载 `configs/<modality>/snapshot_next_frame_supervised.yaml`
- **THEN** `<modality>` 为 `image`、`radar`、`gps`、`lidar` 或 `mmwave` 时配置 MUST 可加载
- **AND** 每个配置 MUST 只要求对应单模态输入字段和标签字段

### Requirement: 多模态 snapshot fusion baseline 配置矩阵
项目 MUST 为合法多模态组合提供 snapshot next-frame supervised fusion 配置入口，至少 MUST 支持五模态 `image_radar_gps_lidar_mmwave`。这些配置 MUST 复用现有 fusion 输入路由和固定模态顺序。

#### Scenario: 加载五模态 snapshot fusion 配置
- **WHEN** 用户加载 `configs/fusion/image_radar_gps_lidar_mmwave_snapshot_next_frame_supervised.yaml`
- **THEN** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型
- **AND** 最终配置 MUST 不包含 distillation 配置块

### Requirement: Snapshot 与历史窗口结果可比较
Snapshot baseline 训练和评估产物 MUST 记录足够 metadata，用于区分 snapshot/no-history 与历史窗口模型，并支撑同 split、同 objective、同 metric 的横向比较。

#### Scenario: 训练产物记录 snapshot 语义
- **WHEN** snapshot baseline 完成训练
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录 `variant: snapshot_next_frame`
- **AND** MUST 记录 `uses_history_window: false`
- **AND** MUST 记录 `uses_temporal_core: false`
- **AND** MUST 记录实际使用的 scene、train/validation CSV 路径和样本数
- **AND** MUST 记录 `split_ratio` 为 `80/20` 或等价结构化字段

#### Scenario: 比较脚本可区分实验族
- **WHEN** 结果汇总工具读取 snapshot baseline 与历史窗口 baseline 的 metrics
- **THEN** 工具 MUST 能从 metadata 区分 `snapshot_next_frame` 和历史窗口配置
- **AND** 工具 MUST 不把两类结果静默合并为同一实验条件
