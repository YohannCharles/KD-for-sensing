# deepverse-dt31-data-generation Specification

## Purpose
TBD - created by archiving change add-deepverse-dt31-data-generation. Update Purpose after archive.
## Requirements
### Requirement: DeepVerse DT31 generator loads scenario parameters
系统 MUST 提供 DeepVerse6G-DT31 数据生成入口，使用 `deepverse.ParameterManager` 和 `deepverse.Dataset` 加载场景参数，并在外部依赖缺失时给出明确错误。

#### Scenario: 缺少 deepverse 包
- **WHEN** 用户运行 DT31 生成脚本且 Python 环境无法导入 `deepverse`
- **THEN** 系统 MUST 失败并提示需要在 `kd_mm_beam` 环境中安装 `deepverse`

#### Scenario: 缺少 DT31 config
- **WHEN** 用户运行 DT31 生成脚本且 `config.m` 不存在
- **THEN** 系统 MUST 失败并提示缺失的绝对路径

#### Scenario: 保存实际生成参数
- **WHEN** DeepVerse dataset object 成功加载
- **THEN** 系统 MUST 将最终参数写入 `used_generation_params.json`

### Requirement: Phase 1 manifest and labels are generated
系统 MUST 从 DeepVerse DT31 dataset object 构建 Phase 1 样本 manifest，并派生 future beam、future trajectory，以及可验证时才启用的 future blockage 标签。

#### Scenario: 有效窗口输出样本
- **WHEN** 某个 UE 的 history 和 future 时间窗口均存在，且 future comm 与 mobility 数据可读取
- **THEN** 系统 MUST 在 `samples.csv` 中写入包含 `sample_id`、`scenario`、`ue_id`、`bs_id`、`t_anchor`、`history_indices`、`future_indices`、`camera_paths`、`lidar_paths`、`radar_feature_history` 和 `split` 的行
- **AND** 系统 MUST 在 `labels.npz` 中写入 beam、trajectory、raw LoS/status、link state、beam gain、valid mask arrays，以及带 `blockage_valid_mask` 的 blockage arrays

#### Scenario: beam 标签由 channel 派生
- **WHEN** future comm sample 包含可用 channel coefficients
- **THEN** 系统 MUST 使用 ULA DFT codebook 计算 beam gain vector，并以最大 gain 的 beam index 作为 future beam label

#### Scenario: blockage 标签由 LoS 派生
- **WHEN** future comm sample 包含 `LoS_status`
- **THEN** 系统 MUST 保存 raw `LoS_status` distribution
- **AND** 系统 MUST 将语义明确的 `LoS_status == 1` 映射为未遮挡 `0`
- **AND** 系统 MUST 仅将语义明确的 NLoS/indirect status 映射为遮挡 `1`
- **AND** 系统 MUST 将 `-1`、缺失值或无法确认语义的 status 标记为 invalid/ignore，而不是默认映射为遮挡正类

#### Scenario: blockage 监督可用性检查
- **WHEN** cache 生成完成且 blockage valid labels 不同时包含未遮挡和遮挡两类，或任一类别低于配置的最低样本数/比例
- **THEN** metadata MUST 将 `blockage.usable` 标记为 `false`
- **AND** sanity report MUST 写入 `blockage.usable: false`、raw LoS/status 分布、valid label 分布和禁用原因
- **AND** 默认训练 objective metadata MUST NOT 将 blockage 作为可训练监督目标

#### Scenario: trajectory 标签由 mobility 派生
- **WHEN** future mobility location 可读取
- **THEN** 系统 MUST 将 future horizon 的二维位置写入 trajectory label，且不得把 clean future location 作为默认输入字段

### Requirement: Radar, weak wireless and noisy position caches are generated
系统 MUST 生成默认可作为后续模型输入的 radar feature history、weak wireless history 和 noisy position history，并避免把 full comm channel 或 clean position 作为默认输入。

#### Scenario: radar feature history 输出
- **WHEN** history radar samples 可读取
- **THEN** 系统 MUST 在 `radar_features.npz` 中写入每个样本的低维历史特征，至少包含 magnitude statistics、phase-difference statistics 和 path count
- **AND** metadata MUST 将 `radar` 记录为默认输入模态

#### Scenario: weak wireless history 输出
- **WHEN** history comm samples 可读取
- **THEN** 系统 MUST 在 `weak_wireless.npz` 中写入每个样本的低维历史特征，至少包含归一化 beam index、max power、top1-top2 power margin 和 beam entropy

#### Scenario: noisy position history 输出
- **WHEN** history mobility location 可读取
- **THEN** 系统 MUST 在 `noisy_position.npz` 中写入带高斯噪声的二维位置历史，并在 metadata 中记录 `position_noise_std`

### Requirement: Split and sanity artifacts are generated
系统 MUST 为 DT31 cache 写出 split、metadata 和 sanity report，记录可复现参数和跳过原因。

#### Scenario: 默认产出无滑窗泄漏的训练和验证划分
- **WHEN** 用户未覆盖 `split_by`
- **THEN** 系统 MUST 按 sequence/segment group 生成 80/20 train/val 划分，并写入 `split.json`
- **AND** train 与 val/test 的 raw history/future time index MUST NOT overlap
- **AND** 对单 UE DT31 数据，validation split MUST 非空且不得通过随机相邻滑窗实现

#### Scenario: 显式按 UE 划分
- **WHEN** 用户设置 `split_by=ue`
- **THEN** 系统 MUST 按 UE 生成 train/val/test 划分，并写入 `split.json`

#### Scenario: 单连续轨迹 fallback
- **WHEN** DT31 dataset object 无法提供可用于 group split 的 scene、pass、object 或 segment id，且只有单条连续轨迹可用
- **THEN** 系统 MUST 使用 contiguous temporal split
- **AND** 系统 MUST purge 或 embargo split 边界附近会导致 history/future raw index 跨 split 重叠的窗口
- **AND** metadata MUST 记录 `split_by: time_contiguous`、边界、embargo span、丢弃窗口数和 split seed

#### Scenario: 显式随机 sample debug split
- **WHEN** 用户显式设置 `split_by=sample_random`
- **THEN** 系统 MAY 按随机 sample 生成 train/val 划分
- **AND** metadata 与 sanity report MUST 标记该 split 的 `leakage_risk: high`
- **AND** 默认配置 MUST NOT 使用 `sample_random`

#### Scenario: 跳过原因统计
- **WHEN** 样本因 mobility、comm、camera、LiDAR、radar、窗口长度或 NaN 标签不可用而被跳过
- **THEN** 系统 MUST 在 `metadata.json` 和 `sanity_report.json` 中记录对应 skip count

#### Scenario: sanity report 输出
- **WHEN** cache 生成完成
- **THEN** 系统 MUST 写出 `sanity_report.json`，至少包含样本数、split 计数、标签分布、artifact 路径、NaN/Inf 检查和缺失模态统计
- **AND** sanity report MUST include radar feature NaN/Inf checks
- **AND** sanity report MUST include raw LoS/status distribution, blockage usability, and raw frame overlap checks across splits

