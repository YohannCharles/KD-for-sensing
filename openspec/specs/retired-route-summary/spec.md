# retired-route-summary Specification

## Purpose
集中记录已折叠退役路线的拒绝边界、历史语义和迁移方向。该能力是 project surface guard，不是当前训练、评估、诊断或数据准备入口。

## Requirements
### Requirement: 折叠退役路线不属于 current support surface
KD、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS coarse anchor、GPS residual、camera residual、geometry residual、BGAM、viewer manifest、仓库级 Gradio viewer、AMR-Net_gps_image、JEPA-MSAC、CRAF、MARF、G2D 和 Multimodal-NF MUST 只作为 retired、historical、migration guard 或防回流说明出现。项目 MUST 不为这些路线提供 current console script、module-only CLI、实体训练 YAML、virtual config、registry current entry、package facade 或本地 runbook shell。

#### Scenario: 旧入口不会回流
- **WHEN** 开发者检查 pyproject、`src/kd_sensing/cli/`、`scripts/`、`configs/` 和 current docs
- **THEN** 上述退役路线 MUST 不作为 current 推荐入口出现
- **AND** 若文本提到旧名称，MUST 明确其 retired、historical、removed、migration guard 或防回流语义

#### Scenario: 旧配置与 override 快速拒绝
- **WHEN** 用户传入旧 Hist、BGAM/Top8/viewer、Raymobtime、legacy KD、AMR-Net_gps_image 或 JEPA-MSAC config path / override / model type
- **THEN** 配置加载、registry 构建或入口检查 MUST fail fast 或返回普通 unknown-name 错误
- **AND** 系统 MUST 不静默迁移到 current workflow

### Requirement: 集中 retired-route guard 取代专用 tombstone 测试
退役路线防回流 MUST 优先由集中 retired-route 清单、参数化测试和 migration guards 维护。项目 MUST 不为每条只剩历史说明的退役路线保留独立 current spec、专用 pytest 文件、兼容 wrapper 或隐藏 CLI。

#### Scenario: 集中测试覆盖拒绝点
- **WHEN** 运行 retired-route focused tests
- **THEN** 测试 MUST 覆盖旧 config path、console fragment、module path、config override、objective、model/loss/dataset/preprocessor registry token 中至少一种拒绝点
- **AND** JEPA-MSAC 与 AMR-Net_gps_image 这类旧 priority workflow MUST 不再拥有专用测试文件

#### Scenario: 保留通用 current helper
- **WHEN** 当前 GPS v2、CSI、JEPA、BeamBench、AMR-Net、geometry prior、safe rerank 或 U-MaskBeamJEPA 仍需要 Top-K、circular metric、geometry helper、reliability metadata 或 full-to-partial stabilization
- **THEN** 实现 MUST 使用当前 owner module 和 current config
- **AND** 实现 MUST 不通过退役 route 的 module path、config path 或 wording 恢复旧 workflow
