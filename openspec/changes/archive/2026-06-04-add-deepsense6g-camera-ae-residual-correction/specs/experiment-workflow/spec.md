## ADDED Requirements

### Requirement: Camera residual staged CLI workflow
项目 MUST 提供 camera residual 分阶段包内 CLI，用于 manifest 构建、Camera AE 训练、AE feature extraction、residual/gate 训练评估、plot 和 compare。所有项目相关 Python 命令 MUST 通过 `conda run -n kd_mm_beam` 运行。

#### Scenario: manifest CLI
- **WHEN** 用户运行 camera residual manifest CLI
- **THEN** CLI MUST 接受 `--config`、`--support-ratio` 和 `--label-space`
- **AND** CLI MUST 输出 camera residual manifest 和 metadata

#### Scenario: AE train CLI
- **WHEN** 用户运行 Camera AE train CLI
- **THEN** CLI MUST 接受 `--config`、`--support-ratio` 和 `--label-space`
- **AND** CLI MUST 保存 checkpoint、metrics 和 reconstruction examples

#### Scenario: AE feature extraction CLI
- **WHEN** 用户运行 AE feature extraction CLI
- **THEN** CLI MUST 接受 `--config`、`--checkpoint`、`--support-ratio` 和 `--label-space`
- **AND** CLI MUST 输出 features、features index 和 manifest with AE

#### Scenario: residual run CLI
- **WHEN** 用户运行 camera residual train/eval CLI
- **THEN** CLI MUST 接受 `--config`、`--support-ratio` 和 `--label-space`
- **AND** CLI MUST 写出 summary、predictions、correction events、candidate recall 和 run metadata

#### Scenario: plot and compare CLI
- **WHEN** 用户运行 camera residual plot 或 compare CLI
- **THEN** plot CLI MUST 从 results dir 生成 figures
- **AND** compare CLI MUST 读取 GPS v2 baseline 与 camera residual summary 并写出 comparison report

### Requirement: Camera residual query leakage guard
camera residual workflow MUST 显式记录并执行 query leakage guard。target query label 只能用于最终 evaluation、predictions、figures 和 report。

#### Scenario: early stopping 不使用 query
- **WHEN** residual/gate 训练启用 early stopping
- **THEN** early stopping MUST 使用 source validation 或 target support 内部 validation
- **AND** target query label MUST NOT 用于模型选择

#### Scenario: query label usage metadata
- **WHEN** camera residual run 完成
- **THEN** run metadata MUST 记录 query label 只用于 evaluation
- **AND** metadata MUST 记录 model selection split
- **AND** metadata MUST 记录 support/query count 和 target scene

#### Scenario: package CLI 边界
- **WHEN** 实现新增 camera residual 入口
- **THEN** 入口 MUST 位于 `src/kd_sensing/cli/`
- **AND** 项目 MUST NOT 新增顶层 `src.*` 运行入口作为兼容包装
