# jepa-msac-reproduction Specification

## Purpose
JEPA-MSAC Scenario 32 paper/workflow reproduction 已退役。本 tombstone 只保留旧 workflow 的拒绝边界和迁移方向；不得重新提供 current CLI、pretraining config、whole-model registry surface、loss、objective、baseline package、focused smoke 或 claim 占位。

## Requirements

### Requirement: JEPA-MSAC retired tombstone
系统 MUST 将 JEPA-MSAC Scenario 32 workflow 视为退役 paper/workflow reproduction。源码、配置、CLI、model registry、loss、objective、tests 和文档账本 MUST 不再把 `jepa_msac` 暴露为 current 可运行入口。

#### Scenario: 旧入口和配置不存在
- **WHEN** 开发者检查 pyproject、包内 CLI、configs、models、losses 和 baseline package
- **THEN** 项目 MUST 不声明 `kd-sensing-run-jepa-msac`
- **AND** 项目 MUST 不保留 `configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml` 作为 current config
- **AND** 项目 MUST 不保留 `kd_sensing.baselines.jepa_msac`、`kd_sensing.models.jepa_msac` 或 `kd_sensing.losses.jepa_msac` 作为 current workflow implementation

#### Scenario: 旧 objective 和 RF mapping 不回流
- **WHEN** 用户加载旧 JEPA-MSAC objective、model type、loss key 或 `workflow.jepa_msac` 配置
- **THEN** 配置加载 MUST 失败并说明 JEPA-MSAC 已退役
- **AND** `rf` MUST 不因 JEPA-MSAC 回流为 canonical modality

#### Scenario: 当前迁移入口清晰
- **WHEN** 用户需要 JEPA 预训练、可视分析或 shortcut benchmark
- **THEN** 文档 SHOULD 指向当前 GPS-conditioned JEPA、JEPA visual analysis、GPS shortcut benchmark 或仍维护的 JEPA downstream workflow
- **AND** 文档 MUST 不要求运行 JEPA-MSAC smoke tests 作为 current health check
