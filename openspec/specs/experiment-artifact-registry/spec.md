# experiment-artifact-registry Specification

## Purpose

定义 MMW T2/baseline 的本地运行产物边界，保证 checkpoint、日志、generated config 和汇总证据不反向污染 tracked recipe 或 package import。

## Requirements

### Requirement: 运行产物不进入 current source surface

训练、评估和 summary 产生的 checkpoint、日志、cache、generated YAML、metrics 与图表 MUST 写入 ignored local output root，且不得被 recipe、registry 或文档当作源码输入。

#### Scenario: 生成本地 evidence

- **WHEN** launcher 或 summary helper 运行
- **THEN** 生成物 MUST 位于 `outputs/` 或显式临时目录
- **AND** tracked recipe MUST 仍能独立解析

### Requirement: 固定 checkpoint provenance

MMW 正式比较 MUST 记录所用 `last.pth`、recipe、seed、split 和 mask identity；这些 provenance 只描述本地 evidence，不形成新的 public API。

#### Scenario: 汇总多 seed 结果

- **WHEN** summary 接受多 seed run
- **THEN** 它 MUST 拒绝缺少必要 provenance 的行
- **AND** 不得混入不同 split 或 mask identity
