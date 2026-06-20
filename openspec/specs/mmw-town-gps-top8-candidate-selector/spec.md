# mmw-town-gps-top8-candidate-selector Specification

## Purpose
记录 MMW Town standalone Top8 candidate manifest 与 BGAM candidate 支撑退役后的防回流边界，同时确保 MMW GPS v2、CSI hardening 和普通 Top-K/normalized-gain 指标继续由当前 owner module 维护。

## Requirements
### Requirement: MMW Town Top8 manifest 与 BGAM candidate 支撑已退役
MMW Town standalone GPS Top8 candidate manifest CLI/config 与 BGAM candidate manifest 支撑不再属于当前支持能力。系统 MUST 删除相关配置、console scripts、包内 CLI、manifest helper 和 focused tests；系统 MUST NOT 通过兼容 stub、thin alias 或 virtual config 恢复这些路径。

#### Scenario: MMW Top8 manifest 入口不存在
- **WHEN** 开发者检查安装入口、配置和包内 CLI
- **THEN** 项目 MUST 不声明 `kd-sensing-prepare-mmw-town-top8-candidate-manifest`
- **AND** 项目 MUST 不保留 `configs/mmw_town_top8_selector.yaml`
- **AND** 包内 MUST 不保留 `kd_sensing.data.mmw_town_topk_candidate_manifest`

#### Scenario: 当前 MMW GPS/CSI 路线不借旧 Top8 支撑复活
- **WHEN** 当前 MMW GPS v2、CSI hardening、group-safe split 或 label-space calibration 需要候选排序、Top-K 或 normalized gain 指标
- **THEN** 实现 MUST 使用当前 MMW/GPS/CSI owner module
- **AND** 实现 MUST NOT 生成旧 Top8 candidate manifest 或导入 BGAM candidate helper
