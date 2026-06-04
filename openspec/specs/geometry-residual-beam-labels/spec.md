# geometry-residual-beam-labels Specification

## Purpose
定义 geometry coarse beam、circular residual beam label 和 clipped residual class 的数据契约，用于把绝对 beam 预测拆解为可解释的几何先验与残差学习目标，并保证 label space 可逆、可诊断且默认不影响 absolute label 路径。
## Requirements
### Requirement: geometry coarse beam 构造
系统 MUST 在启用 geometry-residual label space 时，根据 BS/RSU-centric UE/CAV position 或等价 relative geometry 计算 `geo_angle` 和 `beam_geo`。若真实 codebook 或 beam boundary 不可用，系统 MAY 使用均匀 azimuth quantization，但 MUST 在 metadata 中记录 `beam_geo_source`。

#### Scenario: 从相对位置计算 beam_geo
- **WHEN** 样本包含 UE/CAV position 与 BS/RSU position 或已派生 relative geometry
- **THEN** 系统 MUST 计算 BS/RSU-centric relative position
- **AND** 系统 MUST 计算 azimuth angle
- **AND** 系统 MUST 将 azimuth 量化为 `[0, num_beams)` 内的 `beam_geo`

#### Scenario: geometry 字段不可用
- **WHEN** `label_space.type: geometry_residual` 且样本无法提供 position、relative geometry 或 angle
- **THEN** 若 `label_space.geometry.required: true`，dataset MUST 抛出包含 sample id 和 unavailable reason 的错误
- **AND** 若 `required: false`，dataset MUST 标记 geometry unavailable，并不得伪造 `beam_geo`

### Requirement: circular residual beam label
系统 MUST 支持从 absolute ground-truth beam `beam_abs` 和 geometry coarse beam `beam_geo` 构造 circular residual label。`residual_to_beam(beam_to_residual(beam_abs, beam_geo), beam_geo)` MUST 还原 `beam_abs`。

#### Scenario: residual 可逆
- **WHEN** `beam_abs` 和 `beam_geo` 均在 `[0, num_beams)` 内
- **THEN** `beam_to_residual` MUST 返回符合声明 convention 的 residual
- **AND** `residual_to_beam` MUST 使用同一 convention 还原原始 `beam_abs`

#### Scenario: wrap-around 距离正确
- **WHEN** `num_beams=64` 且比较 beam `0` 与 beam `63`
- **THEN** circular beam distance MUST 等于 `1`
- **AND** residual label MUST 遵守最短环形方向或 metadata 中声明的 full-circular convention

### Requirement: clipped residual class
系统 MUST 支持可选 `max_residual=K` 的 clipped residual class，class id MUST 能映射回 `[-K, K]` 或配置声明的 overflow/ignore 策略。系统 MUST 保留未截断 residual 或 overflow metadata 以支持诊断。

#### Scenario: clipped residual class 范围稳定
- **WHEN** 配置设置 `label_space.max_residual: 8`
- **THEN** dataset MUST 生成范围稳定的 `residual_class`
- **AND** `residual_class_to_delta` MUST 只返回 `[-8, 8]` 内的 delta 或配置声明的 overflow/ignore 标记

#### Scenario: residual 超出 clip 范围
- **WHEN** 某个样本的 circular signed residual 超出 `[-K, K]`
- **THEN** 系统 MUST 按配置将其映射到边界类、overflow 类或 ignore_index
- **AND** metadata MUST 记录 overflow count 和采用的策略

### Requirement: dataset sample 暴露 geometry-residual 字段
当配置 `label_space.type: geometry_residual` 时，dataset sample 或 target provider MUST 暴露 `beam_abs`、`beam_geo`、`beam_residual`、`residual_class`、`geo_angle` 和 `geo_sector` 中可用字段。默认 absolute label path MUST 不要求这些字段存在。

#### Scenario: geometry_residual sample keys
- **WHEN** 用户构建启用 `label_space.type: geometry_residual` 的 dataset
- **THEN** sample MUST 包含 absolute beam label 与可用 geometry/residual 字段
- **AND** runtime metadata MUST 记录 `target_schema=geometry_residual` 或等价标记

#### Scenario: absolute label path 保持兼容
- **WHEN** 用户运行未启用 geometry_residual 的现有训练或评估配置
- **THEN** dataset sample keys MUST 保持既有 absolute beam label 行为
- **AND** 系统 MUST 不要求 position 或 geometry metadata

### Requirement: geometry sector 诊断字段
系统 MUST 支持按 `num_geo_sectors` 将 `geo_angle` 或 `beam_geo` 映射为 `geo_sector`，用于 split 分层、统计和后续误差分析。`geo_sector` 的构造规则 MUST 写入 split 或 run metadata。

#### Scenario: 生成 geo_sector
- **WHEN** 配置设置 `label_space.num_geo_sectors: 8` 且 geometry 可用
- **THEN** 系统 MUST 为样本生成 `[0, 8)` 内的 `geo_sector`
- **AND** metadata MUST 记录 sector boundary 或 beam-to-sector mapping

### Requirement: GPS anchored signed circular residual utilities
系统 MUST 提供 GPS anchored residual correction 所需的 signed circular residual、beam shift、circular window 和 GPS good/bad label 工具，且 MUST 支持 torch 与 numpy 输入。

#### Scenario: signed circular residual wrap-around
- **WHEN** `target=1`、`pred=63`、`num_beams=64`
- **THEN** `signed_circular_residual` MUST 返回 `2`
- **AND** 当 `target=63`、`pred=1`、`num_beams=64` 时 MUST 返回 `-2`

#### Scenario: circular shift beam
- **WHEN** `pred=63`、`delta=2`、`num_beams=64`
- **THEN** `circular_shift_beam` MUST 返回 `1`
- **AND** 返回值 MUST 始终位于 `[0, num_beams)` 范围内

#### Scenario: circular window 处理边界
- **WHEN** `center=0`、`radius=2`、`num_beams=64`
- **THEN** `circular_window` MUST 包含 `62`、`63`、`0`、`1` 和 `2`
- **AND** window MUST 不包含重复 beam id

#### Scenario: GPS good bad label
- **WHEN** circular error 小于配置阈值
- **THEN** good label MUST 为 true 且 bad label MUST 为 false
- **AND** 阈值默认 MUST 为 `4`
- **AND** 该 label MUST 只用于 correction gate 训练和诊断，不得替代最终 DBA 指标

### Requirement: GPS-prior local residual delta class
系统 MUST 支持以 GPS prior top1 为参照的 signed circular residual delta class。该 class MUST 使用 circular signed residual，默认 local delta radius 为 `8`，并提供 overflow class 表示超出 local window 的 residual。

#### Scenario: signed circular residual wrap-around
- **WHEN** `num_beams=64`、`target=1` 且 `gps_pred_top1=63`
- **THEN** `signed_circular_residual(target, gps_pred_top1, num_beams=64)` MUST 返回 `2`
- **AND** 该 residual MUST 能映射到 local delta class

#### Scenario: negative signed residual wrap-around
- **WHEN** `num_beams=64`、`target=63` 且 `gps_pred_top1=1`
- **THEN** `signed_circular_residual(target, gps_pred_top1, num_beams=64)` MUST 返回 `-2`
- **AND** 该 residual MUST 能映射到 local delta class

#### Scenario: residual to delta class
- **WHEN** residual 位于 `[-R, R]`
- **THEN** `residual_to_delta_class(residual, radius=R)` MUST 返回稳定的 class id
- **AND** `delta_class_to_residual(class_id, radius=R)` MUST 还原对应 residual

#### Scenario: overflow class
- **WHEN** residual 的绝对值大于 `R`
- **THEN** `residual_to_delta_class` MUST 返回 overflow class `2R+1`
- **AND** `delta_class_to_residual` 对 overflow class MUST 返回 `None` 或配置声明的 special value
- **AND** diagnostics MUST 能统计 overflow count

