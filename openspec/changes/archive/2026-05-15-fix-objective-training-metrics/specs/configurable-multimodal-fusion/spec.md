## ADDED Requirements

### Requirement: Objective-aware multitask canonical 默认等权
objective-aware fusion canonical 配置 MUST 在 `experiment.objective: multitask` 时默认使用 beam、occlusion 和 position 三个任务等权 loss。该默认值 MUST 应用于所有由 virtual canonical generator 生成的 multitask fusion 配置，包括 all-modalities、strong-only、weak-only 和显式模态 slug。

#### Scenario: 五模态 multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 设置 `experiment.objective: multitask`
- **AND** 解析后的配置 MUST 启用 beam、occlusion 和 position 三类 targets 与 heads
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.beam: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.occlusion: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.position: 1.0`

#### Scenario: strong-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/strong_only_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 只包含 strong modalities `[gps, mmwave]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: weak-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/weak_only_multitask_no_kd.yaml`
- **THEN** 解析后的配置 MUST 只包含 weak modalities `[image, radar, lidar]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: 显式 multitask 权重覆盖
- **WHEN** 用户通过实体 YAML 或命令行覆盖显式设置 `loss.objective.weights.position`
- **THEN** 系统 MUST 使用用户显式配置的 position 权重
- **AND** 该覆盖 MUST 不改变未被覆盖的 beam 和 occlusion 权重

#### Scenario: multitask 权重记录到产物
- **WHEN** 完成 objective-aware multitask 训练
- **THEN** `final_config.yaml` 或等价 runtime metadata MUST 能追溯 beam、occlusion 和 position 的实际 loss 权重
- **AND** epoch log MUST 记录或能派生本次 multitask 总 loss 的权重组成
