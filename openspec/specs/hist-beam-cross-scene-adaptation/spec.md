# hist-beam-cross-scene-adaptation Specification

## Purpose
定义 HiST-Beam 跨场景自适应方法的模型变体、层次化 beam label、adapter/prototype/residual 适配、训练诊断和评估输出契约，确保快速验证中的 source-only、adapter、adapter+prototype 与 full fine-tuning baseline 可配置、可复现并能被 LOSO workflow 汇总比较。
## Requirements
### Requirement: HiST-Beam 研究线已退役
HiST-Beam/Hist 跨场景适配不再属于当前支持能力。系统 MUST 不再提供 HiST-Beam 模型构建、LOSO 执行、target adaptation、prototype、radio/path variant、history-anchor Hist、image-only Hist probe、V7/V8/V9 Hist variant 或 Hist prediction artifact 生成入口。

#### Scenario: Hist 配置不可运行
- **WHEN** 用户尝试运行 HiST-Beam/Hist 配置或 CLI
- **THEN** 系统 MUST 失败、报告入口已退役或缺失对应配置
- **AND** 系统 MUST 不静默迁移到其它当前 workflow

#### Scenario: 当前源码不包含 Hist 模型实现
- **WHEN** 开发者检查当前 `src/kd_sensing/engine` 和 `src/kd_sensing/models`
- **THEN** 当前支持面 MUST 不包含 `hist_beam_*` engine 实现或 `models/fusion/hist_beam.py`
- **AND** registry MUST 不提供 `hist_beam_fusion`

