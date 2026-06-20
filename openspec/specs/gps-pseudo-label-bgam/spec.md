# gps-pseudo-label-bgam Specification

## Purpose
记录 GPS pseudo-history BGAM 输入契约退役后的防回流边界，确保旧 pseudo label、oracle-history upper bound、LiDAR gate 和 BGAM rerank 产物不再作为当前 MMW 或 DeepSense workflow 支撑面。

## Requirements
### Requirement: GPS pseudo-history BGAM workflow 已退役
GPS pseudo-history BGAM workflow 不再属于当前支持能力。系统 MUST 删除或拒绝该 workflow 的配置、manifest 字段契约、BGAM mask source、oracle-history upper-bound 开关、评估产物和 focused tests；系统 MUST NOT 通过新的 facade、virtual config 或兼容 alias 恢复该路径。

#### Scenario: 旧 pseudo-history BGAM 字段不作为当前契约
- **WHEN** 开发者检查当前 dataset、manifest、model forward 和 evaluation report
- **THEN** 当前契约 MUST 不要求 `history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy` 或 `uses_oracle_history_label` 作为 BGAM 输入/输出字段
- **AND** 当前契约 MUST 不要求 `oracle_history_bgam_upper_bound` 作为可运行开关

#### Scenario: 旧配置和值快速失败
- **WHEN** 用户传入包含 `gps_lidar_bgam`、`lidar_bgam`、`bgam` 或 viewer/BGAM 退役字段的 config/override
- **THEN** migration guard MUST 早失败
- **AND** 错误信息 MUST 说明该 BGAM 支持面已退役且不再提供兼容入口

#### Scenario: 当前 GPS 诊断不复用 BGAM 命名
- **WHEN** 当前 MMW GPS v2、CSI hardening 或 JEPA/GPS shortcut benchmark 需要 GPS 预测、历史上下文或 calibration metadata
- **THEN** 实现 MUST 使用当前 owner module 和当前字段契约
- **AND** 实现 MUST NOT 通过 pseudo-history BGAM module path、输出目录或文档命令暴露能力
