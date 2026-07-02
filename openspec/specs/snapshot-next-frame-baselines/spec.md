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

### Requirement: Snapshot workflow metadata
训练、验证和评估流程 MUST 在 snapshot next-frame baseline 的运行产物中记录该实验的无历史窗口语义。metadata MUST 足以让结果汇总工具区分 snapshot baseline 与历史窗口 baseline。

#### Scenario: 训练记录 snapshot metadata
- **WHEN** 用户训练 snapshot next-frame baseline
- **THEN** 运行 metadata MUST 记录 `variant: snapshot_next_frame`
- **AND** MUST 记录 `seq_len: 1` 和 `num_pred: 1`
- **AND** MUST 记录 `uses_history_window: false`
- **AND** MUST 记录 `uses_temporal_core: false`

#### Scenario: 评估报告记录 snapshot metadata
- **WHEN** 用户评估 snapshot next-frame baseline checkpoint
- **THEN** 评估报告 MUST 包含 checkpoint 或配置中的 snapshot metadata
- **AND** 报告 MUST 记录 enabled modalities、objective、scene、train/validation split CSV 和样本数
- **AND** 报告 MUST 标记 validation split 是 80/20 协议中的验证集合

### Requirement: Snapshot smoke workflow
项目 MUST 提供可通过统一训练入口运行的 snapshot smoke workflow。该 workflow MUST 使用 `conda run -n kd_mm_beam` 运行测试、训练或评估命令。

#### Scenario: 单模态 snapshot smoke test
- **WHEN** 开发者运行单模态 snapshot 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、validation 和 checkpoint 保存
- **AND** 日志中的模型配置 MUST 显示无 GRU representation core

#### Scenario: 多模态 snapshot smoke test
- **WHEN** 开发者运行五模态 snapshot fusion 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 通过现有 fusion batch preparation 构造启用模态输入
- **AND** forward 输出 MUST 与 `num_pred=1` 的 labels 对齐
- **AND** 训练流程 MUST 不加载 teacher checkpoint

### Requirement: Snapshot 与历史窗口比较输出
实验工作流 MUST 允许用户在同一 Scenario 31 和同一 objective 下比较 snapshot baseline 与历史窗口 baseline。比较输出 MUST 明确展示实验变体和 split 协议，避免把不同时间上下文或不同窗口生成口径的结果混为同一条件。

#### Scenario: 记录 split 协议差异
- **WHEN** 用户对同一模态运行 snapshot baseline 和历史窗口 baseline
- **THEN** 两次运行的 metadata MUST 记录各自 train/validation CSV 路径和样本数
- **AND** 如果 CSV 路径或样本数不同，比较工具或文档 MUST 要求用户将其视为不同数据口径

#### Scenario: 结果表包含时间上下文
- **WHEN** 工具汇总 snapshot 与历史窗口结果
- **THEN** 表格或 JSON 输出 MUST 包含 `variant`、`seq_len`、`num_pred`、`uses_temporal_core` 和 `split_protocol`
- **AND** 模态强弱排序 MUST 能按这些字段分组计算

### Requirement: 序列预处理输出 future 位置列
序列 CSV 预处理 MUST 支持显式输出 future GPS/BS GPS 位置目标列。该开关启用时，每个合法窗口 MUST 在保留现有 `beam` 和 `future_beam` 列的同时，输出与预测 horizon 对齐的 `future_gps` 和 `future_bs_gps` 列。

#### Scenario: 生成 future GPS target 列
- **WHEN** 用户运行序列预处理并设置 `include_position_targets: true`
- **THEN** 输出 CSV MUST 包含 `future_gps1..future_gpsN`
- **AND** 输出 CSV MUST 包含 `future_bs_gps1..future_bs_gpsN`
- **AND** `N` MUST 等于配置的预测长度 `out_len`

#### Scenario: 不启用时保持旧 CSV 结构
- **WHEN** 用户运行序列预处理但未启用 `include_position_targets`
- **THEN** 输出 CSV MUST 保持现有列结构兼容
- **AND** 旧的 beam-only dataset MUST 能继续读取该 CSV

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

### Requirement: Snapshot fusion 配置
Fusion 配置体系 MUST 支持 snapshot next-frame no-KD baseline。该 baseline MUST 使用现有 `experiment.task: fusion` 输入路由、现有 `modalities` 标准化和现有 fusion batch 准备，但模型必须为无时序 snapshot fusion 模型。

#### Scenario: 五模态 snapshot fusion
- **WHEN** 用户加载五模态 snapshot fusion 配置
- **THEN** 配置 MUST 启用 `image`、`radar`、`gps`、`lidar` 和 `mmwave`
- **AND** dataset MUST 按现有 fusion 模态选择逻辑只读取启用模态
- **AND** 模型 MUST 对当前帧的五个模态表示执行融合
- **AND** 模型 MUST 输出 `[B, 1, num_classes]` logits

#### Scenario: 任意合法多模态 snapshot fusion
- **WHEN** 用户加载 `configs/fusion/<slug>_snapshot_next_frame_supervised.yaml` 且 `<slug>` 是两个到五个合法模态组成的 canonical slug
- **THEN** 系统 MUST 使用 `<slug>` 表示的模态集合构建 snapshot fusion
- **AND** forward MUST 只要求该模态集合对应的输入张量
- **AND** 未启用模态缺失不得阻止该配置运行

### Requirement: Snapshot fusion 不依赖 legacy fusion GRU
Snapshot fusion baseline MUST 不使用 `fusion_teacher`、`fusion_student` 的 GRU 路线作为主模型。训练主模型 MUST 是无时序 snapshot 模型。

#### Scenario: supervised snapshot fusion 主模型
- **WHEN** 用户训练 snapshot fusion supervised 配置
- **THEN** 可训练主模型 MUST 为无时序 snapshot 模型
- **AND** 训练流程 MUST 不构建 frozen teacher checkpoint
- **AND** 最终配置 MUST 不包含 `distillation`

#### Scenario: legacy fusion GRU 不参与 snapshot forward
- **WHEN** snapshot fusion 模型执行 forward
- **THEN** forward 路径 MUST 不调用 legacy `fusion_teacher` 或 `fusion_student` 的 GRU 层
- **AND** output diagnostics 或 final config MUST 标记 `uses_temporal_core: false`

### Requirement: Snapshot canonical 配置解析
配置加载流程 MUST 能识别并生成 snapshot next-frame baseline 配置。可生成路径 MUST 包含单模态 `configs/<modality>/snapshot_next_frame_no_kd.yaml` 和 fusion `configs/fusion/<slug>_snapshot_next_frame_no_kd.yaml`。

#### Scenario: 生成单模态 snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/gps/snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 GPS snapshot 配置
- **AND** 最终配置 MUST 设置 `experiment.task: gps`
- **AND** 最终配置 MUST 设置 `data.dataset.seq_len: 1` 和 `data.dataset.num_pred: 1`
- **AND** 最终配置 MUST 设置 `data.dataset.train_csv_name: train_seqs_SNAPSHOT_NEXT_FRAME.csv`
- **AND** 最终配置 MUST 设置 `data.dataset.val_csv_name: val_seqs_SNAPSHOT_NEXT_FRAME.csv` 或等价 validation CSV 字段
- **AND** 最终配置 MUST 构建 `snapshot_frame` core

#### Scenario: 生成 fusion snapshot 配置
- **WHEN** 用户加载缺失但合法的 `configs/fusion/gps_mmwave_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 生成可用于训练和评估的 fusion snapshot 配置
- **AND** 最终配置 MUST 设置启用模态为 `["gps", "mmwave"]`
- **AND** 最终配置 MUST 设置 `experiment.task: fusion`
- **AND** 最终配置 MUST 构建无时序 snapshot fusion 模型

#### Scenario: 拒绝非法 snapshot slug
- **WHEN** 用户加载应被拒绝的非法路径 `configs/fusion/mmwave_gps_snapshot_next_frame_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该路径
- **AND** 错误信息 MUST 提示 canonical slug 为 `gps_mmwave`

### Requirement: Snapshot 配置生成语义
生成的 snapshot 配置 MUST 明确覆盖历史窗口默认值，并保持实体 YAML 优先、命令行覆盖后应用和 schema 校验流程一致。

#### Scenario: 覆盖历史窗口默认值
- **WHEN** 系统生成任一 snapshot 配置
- **THEN** 生成配置 MUST 覆盖默认 `seq_len=8` 和 `num_pred=3`
- **AND** 生成配置 MUST 覆盖默认 GRU representation core
- **AND** 生成配置 MUST 覆盖默认历史窗口 CSV 为 snapshot 专用 train/validation CSV
- **AND** 生成配置 MUST 设置 `output.run_name` 包含 `snapshot_next_frame_no_kd`

#### Scenario: 实体 snapshot 配置优先
- **WHEN** 用户加载磁盘上存在的 snapshot YAML
- **THEN** 配置加载流程 MUST 使用实体 YAML 内容
- **AND** virtual snapshot 规则 MUST 不覆盖同名实体配置

#### Scenario: 命令行覆盖仍生效
- **WHEN** 用户加载 snapshot virtual 配置并传入覆盖项 `training.epochs=1`
- **THEN** 最终配置 MUST 使用 `training.epochs: 1`
- **AND** snapshot 必需字段 MUST 继续满足 `seq_len=1`、`num_pred=1` 和无时序 core 契约，除非用户显式退出 snapshot 变体并通过校验
