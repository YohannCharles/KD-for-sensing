# deepsense6g-gps-top8-candidate-selector Specification

## Purpose
记录 DeepSense6G standalone Top8 selector 与 BGAM-only TopK candidate 支撑退役后的边界，同时保留普通 Top-K 指标、64-beam circular metric 和当前 GPS v2/CSI/JEPA 路线所需的通用能力。

## Requirements
### Requirement: DeepSense6G Top8 selector 与 BGAM-only TopK 支撑已退役
DeepSense6G standalone Top8 selector 训练/plot/compare workflow 与 BGAM-only TopK candidate manifest、dataset、loss 支撑不再属于当前支持能力。系统 MUST 删除相关配置、console scripts、包内 CLI、engine、model、data helper、loss helper 和 focused tests；系统 MUST NOT 通过兼容 stub、thin alias 或 virtual config 恢复这些路径。

#### Scenario: Top8 selector 和 candidate manifest 入口不存在
- **WHEN** 开发者检查配置、安装入口和包内 CLI
- **THEN** 项目 MUST 不声明 DeepSense6G Top8 selector 或 Top8 candidate manifest 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_top8_selector.yaml`
- **AND** 包内 MUST 不保留 `prepare_deepsense6g_top8_candidate_manifest` 入口模块

#### Scenario: BGAM-only TopK 支撑模块不存在
- **WHEN** 开发者检查 source tree 和 import surface
- **THEN** 项目 MUST 不保留 `kd_sensing.data.deepsense6g_topk_candidate_manifest`
- **AND** 项目 MUST 不保留 `kd_sensing.losses.topk_candidate_losses`
- **AND** 项目 MUST 不保留 Top8 selector 专属 model、engine 或 focused tests

### Requirement: 普通 Top-K 与 circular metric 能力保持当前可用
Top8 selector/BGAM-only 支撑退役 MUST NOT 删除普通 Top-K accuracy、candidate ranking 诊断、64-beam circular distance 或当前 GPS v2/CSI 路线需要的 label-space metric。系统 MUST 将这些能力保留在当前 owner module 中，而不是旧 selector 或 BGAM module path 中。

#### Scenario: 当前指标不依赖旧 selector 模块
- **WHEN** 当前训练、评估、CSI ranking 或 GPS v2 诊断计算 Top-K/circular 指标
- **THEN** 实现 MUST 使用当前通用 metric helper 或当前 owner module
- **AND** 实现 MUST NOT 导入旧 Top8 selector、TopK candidate manifest 或 BGAM-only loss module
