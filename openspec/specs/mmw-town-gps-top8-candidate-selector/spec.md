# mmw-town-gps-top8-candidate-selector Specification

## Purpose
TBD - created by archiving change add-gps-pseudo-label-bgam. Update Purpose after archive.
## Requirements
### Requirement: MMW Town GPS Top8 candidate manifest workflow
系统 MUST 提供显式 opt-in 的 MMW Town GPS Top8 candidate manifest workflow。该 workflow MUST 默认覆盖 `Town10_crossroad_seed24`、`Town10_skybridge_seed24`、`Town10_curvyroad_seed42` 和 `Town10_Hroad_seed42`，使用 `mapping_enabled`、`num_beams=64`、MMW GPS v2 frozen logits、circular beam distance 和 MMW prepared split 中的 future beam label。

#### Scenario: 默认 MMW Top8 配置
- **WHEN** 用户运行 MMW Town Top8 默认配置
- **THEN** 系统 MUST 读取 `configs/mmw_town_top8_selector.yaml`
- **AND** 系统 MUST 使用 MMW `dataset/MMW/sunny` Town10 scenes
- **AND** 系统 MUST 默认使用 `mapping_enabled`
- **AND** 输出 MUST 写入 `outputs/analysis/mmw_town_top8_selector/mapping_enabled/`

#### Scenario: 从 MMW GPS v2 logits 构建 Top8
- **WHEN** MMW GPS v2 logits 和 logits index 可用
- **THEN** Top8 candidate beams MUST 从 logits 重新计算
- **AND** 系统 MUST NOT 从 `predictions.csv` 的 Top5 字段截断推导 Top8
- **AND** manifest MUST 保留 `gps_logits_row_index`、`sample_id`、`scene`、`timestamp/frame_id`、`support_query_role` 和 `top8_manifest_row_index`

#### Scenario: scene-specific label-space 校验
- **WHEN** GPS logits index、predictions、support manifest 或 Top8 manifest 包含 MMW scene-specific mapping fingerprint
- **THEN** 系统 MUST 按 row 的 scene 解析 expected fingerprint
- **AND** fingerprint 不一致时 MUST 早失败
- **AND** 错误信息 MUST 指出 artifact 路径、scene、source fingerprint 和 expected fingerprint

#### Scenario: 输出 MMW normalized gain 诊断字段
- **WHEN** prepared row 提供 future beam power path
- **THEN** Top8 manifest SHOULD 计算 `gps_normalized_gain`、candidate normalized gain 和 `top8_oracle_normalized_gain`
- **AND** 若 label mapping 启用，系统 MUST 用 scene-specific inverse mapping 将 mapped beam 转回 raw beam power index
- **AND** normalized gain 缺失 MUST 为空值并在 metadata 中可诊断，不得阻塞 Top8 manifest 构建

