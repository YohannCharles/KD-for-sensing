# ieee-11282996-gps-image-reproduction Specification

## Purpose
AMR-Net_gps_image / IEEE `11282996` source-audit 和 Scenario 23 local-substitute workflow 已退役。本 tombstone 只保留 metadata conflict 背景、旧入口拒绝边界和迁移方向；不得重新提供 current CLI、实体配置、mock metrics、paper model group 或 claim 占位。

## Requirements

### Requirement: AMR-Net_gps_image retired tombstone
系统 MUST 将 AMR-Net_gps_image / IEEE `11282996` 视为退役 source-audit/local-substitute workflow。项目 MUST NOT 将旧 CLI、config、baseline package、paper model groups、report writer、runtime output root 或 claim row 暴露为 current entry point。

#### Scenario: 旧入口和配置被拒绝
- **WHEN** 开发者检查 pyproject、包内 CLI、configs 和 baselines package
- **THEN** 项目 MUST 不声明 `kd-sensing-run-amr-net-gps-image`
- **AND** 项目 MUST 不保留 `configs/baselines/amr_net_gps_image.yaml` 作为 current config
- **AND** 项目 MUST 不保留 `kd_sensing.baselines.amr_net_gps_image` 作为 current workflow package

#### Scenario: 历史 metadata conflict 只作为背景
- **WHEN** 文档提到 IEEE `11282996`、AMR-Net_gps_image 或 Scenario 23 local substitute
- **THEN** 文档 MUST 将其标记为 retired、historical、blocked background 或 tombstone
- **AND** 文档 MUST 不提供 current 运行命令、mock metric 命令或 official reproduction claim

#### Scenario: 当前迁移入口清晰
- **WHEN** 用户需要 GPS+Image 或 vision-position 对照
- **THEN** 文档 SHOULD 指向当前 Vision-Position baseline suite、BeamBench/Arnold22 Camera AE+GPS Direct 或其它 current baseline/control
- **AND** migration guard 错误信息 SHOULD 说明 AMR-Net_gps_image 已退役
