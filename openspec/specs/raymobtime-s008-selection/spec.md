# raymobtime-s008-selection Specification

## Purpose
定义 Raymobtime s008 数据审计、cache 构建、任务语义和训练评估入口，确保 current snapshot beam selection 流程边界明确。
## Requirements
### Requirement: Raymobtime s008 数据审计与 cache 构建
系统 MUST 提供 Raymobtime s008 专用的配置驱动预处理能力，用于审计本地数据、构建 snapshot 样本索引、标准化 beam/LOS/link 标签，并从 ray-tracing zip 提取 path-level 特征。预处理能力 MUST 通过包内 PREPROCESSORS 和 `kd-sensing-preprocess` 或等价包内 CLI 暴露，不得要求用户运行绕过 `src/kd_sensing` 包结构的顶层旧脚本。

#### Scenario: 审计必需文件
- **WHEN** 用户对 Raymobtime s008 数据根目录运行审计预处理
- **THEN** 系统 MUST 检查 `baseline_data/beam_output`、`baseline_data/coord_input`、`baseline_data/lidar_input`、`baseline_data/image_v2_input`、`raw_data/CoordVehiclesRxPerScene_s008.csv` 和 `raw_data/ray_tracing_data_s008_carrier60GHz.zip`
- **AND** 缺失任一必需路径时 MUST 报出包含缺失路径的清晰错误

#### Scenario: 输出审计摘要
- **WHEN** Raymobtime s008 审计完成
- **THEN** 系统 MUST 在配置指定的输出目录写出 npz keys/shape/dtype 摘要、CSV 行数与关键列分布、LOS 分布、坐标范围和 episode/scene/vehicle 唯一值统计
- **AND** 这些审计输出 MUST 默认位于 `outputs/raymobtime_s008/audit` 或用户配置的 ignored 输出目录

#### Scenario: 构建 snapshot index
- **WHEN** 用户运行 Raymobtime s008 index 构建预处理
- **THEN** 系统 MUST 从 `CoordVehiclesRxPerScene_s008.csv` 读取 `Val`、`EpisodeID`、`SceneID`、`VehicleArrayID`、`VehicleName`、`x`、`y`、`z` 和 `LOS`
- **AND** 系统 MUST 只保留 `Val == "V"` 的 receiver 样本
- **AND** 每行 MUST 使用 `EpisodeID`、`SceneID` 和 `VehicleArrayID` 构造稳定 `sample_id`

#### Scenario: 标准化 beam 标签格式
- **WHEN** cache builder 读取 `baseline_data/beam_output`
- **THEN** 系统 MUST 支持 `[N]` beam class、`[N, 2]` Tx/Rx beam pair 和 `[N, Tx, Rx]` beam score matrix 三类输入
- **AND** 系统 MUST 统一输出 `beam_label`、`beam_tx`、`beam_rx`、`num_beam_classes`、`num_tx_beams` 和 `num_rx_beams`

#### Scenario: 解析官方 HDF5 ray tracing
- **WHEN** `raw_data/ray_tracing_data_s008_carrier60GHz.zip` 包含官方 `.hdf5` ray-tracing 条目
- **THEN** 系统 MUST 读取 `.hdf5` 条目并从 path-level 数据生成 canonical ray rows
- **AND** canonical ray rows MUST 能与 snapshot index 的 `EpisodeID`、`SceneID`、`VehicleArrayID` 或稳定 `sample_id` 对齐
- **AND** canonical ray rows MUST 至少提供可用于 `max_power_dbm`、`mean_power_dbm_valid`、`sum_power_linear`、`num_valid_rays` 和 link target 的 power 字段
- **AND** 当运行环境缺少必需 HDF5 读取依赖时，预处理 MUST 报出包含依赖名称和 Raymobtime s008 HDF5 解析上下文的清晰错误

#### Scenario: ray 特征禁止 LOS 泄漏
- **WHEN** 系统从 ray-tracing zip 生成 Raymobtime s008 ray feature cache
- **THEN** 模型输入 cache MUST 使用不包含 LOS flag 的 `ray_features_no_los`
- **AND** 包含 LOS flag 的 `ray_features_with_los` MUST 只用于审计或数据质量校验
- **AND** link quality target MUST 与 ray 输入特征分开保存

#### Scenario: link target 质量门禁
- **WHEN** 系统生成 Raymobtime s008 `link_quality` cache
- **THEN** cache metadata MUST 记录 matched ray path 数、unmatched 样本数、fallback link target 数、link target min/mean/max/std 和 fallback 占比
- **AND** 当全部样本缺失 ray path、`link_quality` 全部等于 fallback 值或训练 split 的 link target 标准差为 0 时，预处理 MUST 拒绝生成可训练 cache
- **AND** 错误信息 MUST 指向 ray-tracing HDF5 解析、sample_id 对齐或数据根路径配置问题

### Requirement: Raymobtime s008 snapshot dataset 契约
系统 MUST 注册 `raymobtime_s008` dataset type，用于当前快照 beam selection。该 dataset MUST 返回 flat dict batch 字段以接入现有 DataLoader/runtime，并 MUST 保持 Raymobtime s008 的当前快照语义，不暴露 history、future、horizon、beam tracking 或 sequence transition 字段。

#### Scenario: dataset 返回 flat snapshot batch
- **WHEN** 用户构建 `data.dataset.type: raymobtime_s008` 的 train/test dataset
- **THEN** 每个样本 MUST 返回启用模态对应的 flat 输入字段，字段名来自中心化模态契约
- **AND** 当前 beam label MUST 返回为 `target_beam`
- **AND** LOS 与 link targets MUST 分别返回为 `los_label` 和 `link_quality`
- **AND** metadata MUST 至少包含 `sample_id`、`EpisodeID`、`SceneID`、`VehicleArrayID`、`valid_index` 和 `split`

#### Scenario: snapshot 张量形状
- **WHEN** Raymobtime s008 dataset 返回启用的 `coord`、`image`、`lidar` 或 `ray` 输入
- **THEN** 每个输入 MUST 保留单步 snapshot 维度
- **AND** `coord` MUST 具有 `[1, F_coord]` 语义
- **AND** `image` MUST 具有 `[1, 3, H, W]` 语义
- **AND** `lidar` MUST 具有 `[1, C, D, H, W]` 3D occupancy grid 语义
- **AND** `ray` MUST 具有 `[1, F_ray]` 语义

#### Scenario: 拒绝时序字段
- **WHEN** Raymobtime s008 dataset 或 cache 中出现需要历史窗口或未来 horizon 的配置字段
- **THEN** 系统 MUST 拒绝该配置或忽略非 Raymobtime 字段并发出清晰错误
- **AND** 错误信息 MUST 指出 s008 仅支持 current snapshot beam selection

#### Scenario: split metadata 可复现
- **WHEN** Raymobtime s008 cache 构建完成
- **THEN** 系统 MUST 写出 train/validation/test 或配置指定 split 的 index 文件和 split metadata
- **AND** metadata MUST 记录 split seed、样本数、beam label 分布、LOS 分布、link target 口径和输入数据 fingerprint

### Requirement: Raymobtime s008 current beam selection 模型
系统 MUST 提供无时序 Raymobtime s008 snapshot 模型，用于单模态和多模态 current beam selection、LOS 分类和 link quality 回归。模型 MUST 不包含 GRU、RNN、LSTM、TCN 或跨时间 self-attention。

#### Scenario: simple concat 多任务模型输出
- **WHEN** 用户构建 `simple_concat_multitask_selection` 模型并启用任意非空 Raymobtime 模态集合
- **THEN** 模型 MUST 对每个启用模态编码当前 snapshot 特征
- **AND** 模型 MUST 输出 beam logits `[B, 1, num_beam_classes]`
- **AND** 模型 MUST 输出 LOS logits `[B, 1]`
- **AND** 模型 MUST 输出 link prediction `[B, 1]`

#### Scenario: task-aware gate 输出
- **WHEN** 用户构建 `task_aware_gated_multitask_selection` 模型
- **THEN** 模型 MUST 为 `beam_selection`、`los` 和 `link_quality` 分别生成 `[B, K]` gate
- **AND** diagnostics MUST 携带 gate 对应的模态名称和任务名称
- **AND** gate 权重 MUST 在每个任务内部按启用模态归一化

#### Scenario: image encoder 复用现有 RGB/ImageNet 契约
- **WHEN** Raymobtime s008 snapshot 模型启用 `image` 模态
- **THEN** 模型 MUST 使用现有 `resnet18_imagenet_rgb` encoder
- **AND** dataset/runtime MUST 向该 encoder 提供 `[B, 1, 3, 224, 224]` 的 `rgb_imagenet` 输入
- **AND** 模型 MUST 不使用 Raymobtime 专用轻量 image CNN

#### Scenario: LiDAR 3D occupancy encoder
- **WHEN** Raymobtime s008 snapshot 模型启用 `lidar` 模态
- **THEN** 模型 MUST 使用专用 `raymobtime_lidar_3d_cnn` encoder
- **AND** encoder MUST 按 3D occupancy grid -> 3D Conv Stem -> 3D Residual Blocks -> Channel Attention -> Global AvgPool + Global MaxPool -> MLP Projection Head -> LiDAR embedding 的结构处理输入
- **AND** encoder MUST 接受 `[B, 1, C, D, H, W]` 当前快照输入并返回 `[B, 1, D_model]` embedding

#### Scenario: 拒绝时间维大于一
- **WHEN** Raymobtime s008 snapshot 模型收到时间维大于 1 的输入
- **THEN** 模型 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 Raymobtime s008 模型只接受当前 snapshot

### Requirement: Raymobtime s008 评估指标
系统 MUST 为 Raymobtime s008 输出 current beam selection、LOS 分类和 link quality 回归指标。指标 MUST 按当前 `experiment.objective` 区分正式 validation metrics 与辅助诊断，正式指标 MUST 记录在训练日志、评估报告和分析输出中，并可用于 early stopping 或实验汇总。

#### Scenario: beam selection 指标
- **WHEN** Raymobtime s008 验证或评估 current beam selection objective
- **THEN** metrics MUST 包含 `beam_top1`、`beam_top3`、`beam_top5` 和 `beam_dba_current`
- **AND** validation metrics MUST 包含 `val_beam_top1`、`val_beam_top3`、`val_beam_top5` 和 `val_beam_dba`
- **AND** 这些指标 MUST 基于当前最优 beam class 计算，不得计算或暴露 future-only `val_adba`

#### Scenario: LOS 分类指标
- **WHEN** Raymobtime s008 验证或评估 current LOS classification objective 或 selection multitask objective
- **THEN** metrics MUST 包含 `los_accuracy`、`los_f1` 和 `los_auc`
- **AND** 当当前 split 只有单个 LOS 类别导致 AUC 不可定义时，系统 MUST 在 metrics 中记录 AUC 不可用状态而不是静默返回错误数值

#### Scenario: link quality 指标
- **WHEN** Raymobtime s008 验证或评估 current link quality objective 或 selection multitask objective
- **THEN** metrics MUST 包含 `link_mae`、`link_rmse` 和 `link_r2`
- **AND** metrics metadata MUST 记录 link target 名称、单位、聚合方式和 cache 质量摘要

#### Scenario: 单任务正式指标隔离
- **WHEN** Raymobtime s008 验证或评估 `current_beam_selection`、`current_los_classification` 或 `current_link_quality` 单任务 objective
- **THEN** `available_metrics` MUST 只包含当前 objective 的正式指标和 `val_loss`
- **AND** 非当前 objective 的 head 输出 MAY 被记录在内部诊断 payload 中
- **AND** 非当前 objective 的 head 输出 MUST NOT 写入正式 `val_*` metrics、history 字段或 TensorBoard 主指标 tag

#### Scenario: selection multitask 指标
- **WHEN** Raymobtime s008 验证或评估 `selection_multitask` objective
- **THEN** metrics MUST 同时包含 beam Top-K、`beam_dba_current`、LOS accuracy/F1/AUC、link MAE/RMSE/R2 和 `val_selection_multitask_loss`
- **AND** `available_metrics` MUST 包含这些字段，以支持 early stopping 校验和结果汇总
