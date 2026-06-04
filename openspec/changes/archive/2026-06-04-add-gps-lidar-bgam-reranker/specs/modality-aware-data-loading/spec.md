## ADDED Requirements

### Requirement: GPS+LiDAR BGAM 按需模态加载
GPS+LiDAR BGAM dataset MUST 按配置和 manifest availability 加载 GPS prior、TopK candidates 和 LiDAR 输入。未启用 LiDAR、image、camera AE 或 radar 时，dataset MUST 不触发对应模态 IO；GPS-only ablation MUST 不读取 LiDAR 点云或 BEV cache。

#### Scenario: gps_only 不读取 LiDAR
- **WHEN** 配置运行 `gps_only` ablation
- **THEN** dataset MUST 只读取 BGAM manifest 中的 GPS prior、candidate beams/probs 和 label/evaluation metadata
- **AND** dataset MUST NOT 读取 raw LiDAR point cloud、LiDAR BEV cache、image、camera AE 或 radar feature

#### Scenario: BGAM ablation 按需读取 LiDAR
- **WHEN** 配置运行包含 BGAM 或 LiDAR 的 ablation
- **THEN** dataset MUST 读取当前样本所需的 `lidar_bev_cache_path` 或 `lidar_path`
- **AND** dataset MUST NOT 读取未启用的 image、camera AE 或 radar feature
- **AND** LiDAR 读取 MUST 发生在取样阶段而不是 dataset 初始化阶段

#### Scenario: LiDAR 缺失时记录 skipped reason
- **WHEN** 配置启用 LiDAR ablation 但 manifest 行缺少 LiDAR path 或文件不存在
- **THEN** 系统 MUST 早失败或按配置跳过该 ablation
- **AND** summary/run metadata MUST 写入 `skipped_reason`、缺失字段和受影响样本数

### Requirement: GPS+LiDAR BGAM 防泄漏数据边界
GPS+LiDAR BGAM 数据构建 MUST 区分训练输入、loss label 和最终评价字段。future ground-truth beam label MUST 只作为 loss/evaluation target；target query rows MUST 不参与 normalization fit、mask construction、early stopping 或 checkpoint selection。

#### Scenario: target label 不进入模型输入
- **WHEN** dataset 返回一个训练或评估样本
- **THEN** `gt_beam` 或 `target_label` MUST 单独作为 label 字段返回
- **AND** 模型输入字段 MUST 不包含由 target label 派生的 BGAM mask、AoD prior、candidate probability 或 LiDAR feature

#### Scenario: query 不参与 normalizer fit
- **WHEN** BGAM dataset 或 runner fit GPS/LiDAR/candidate normalizer
- **THEN** fit rows MUST 只来自 source train、target support 或 target support internal train split
- **AND** target query rows MUST 只使用已 fit 的 normalizer transform
- **AND** metadata MUST 记录 `query_label_used_for_training=false`

### Requirement: GPS+LiDAR BGAM manifest column mapping
BGAM manifest loader MUST 支持配置化字段名映射，以兼容 Top8 manifest、DeepSense6G sequence CSV 和用户提供的 GPS+LiDAR manifest。字段映射 MUST 输出统一内部字段，并 MUST 在缺失必要字段时给出清晰错误。

#### Scenario: local coordinate columns
- **WHEN** manifest 提供 local coordinate columns
- **THEN** loader MUST 按配置映射为 `user_x`、`user_y`、`rsu_x`、`rsu_y` 和 `rsu_yaw`
- **AND** loader MUST 使用这些字段生成 `theta_gps` 和 `distance_to_rsu`

#### Scenario: GPS logits/probs columns
- **WHEN** manifest 提供 `gps_prob_0` 到 `gps_prob_63` 或 `gps_logits_path`
- **THEN** loader MUST 读取或构造 `[64]` GPS prior tensor
- **AND** loader MUST 从该 prior 生成 TopK candidates 或校验与 manifest candidates 一致
