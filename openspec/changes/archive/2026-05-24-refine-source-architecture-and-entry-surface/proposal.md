## Why

当前源码表面积已经完成一次收敛，但仍有少数大模块继续承载多类职责，配置矩阵和研究入口也需要更明确的生命周期规则。现在适合在不触碰 `dataset/`、`outputs/`、`logs/` 等本地数据和产物的前提下，进一步降低后续实验扩展的维护成本。

## What Changes

- 拆分源码大模块的职责边界，优先覆盖 viewer 工具、Raymobtime s008 预处理、互补性分析和 CSI 模型相关实现。
- 为 `tools/visualization`、`src/kd_sensing/preprocessing`、`src/kd_sensing/diagnostics` 和 `src/kd_sensing/models` 中的大文件建立可审计的模块化目标和回归检查。
- 继续推进高级 fusion、CRAF、MARF、CSI/GPS/mmWave 组合配置的 recipe 化，删除实体 YAML 前必须完成关键语义等价检查。
- 梳理 `scripts/`、`tools/analysis/`、`tools/visualization/` 的入口生命周期，区分包内 CLI、薄 alias、研究诊断脚本、数据准备脚本和 viewer 支持文件。
- 强化表面积 inventory 与架构边界测试，使新增实体 YAML、重复入口或大文件职责回流都需要 OpenSpec 说明。
- 不清理、不移动、不压缩真实数据集、本地运行产物、checkpoint、cache 或日志；这些内容仍只受现有 `.gitignore` 和产物边界约束。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 明确源码大模块拆分目标、入口生命周期分类、表面积回归检查和“不触碰数据/产物”的架构边界。
- `canonical-config-resolution`: 扩展高级配置 recipe 化和实体 YAML 删除前的等价校验要求，覆盖保留矩阵的二次瘦身流程。
- `experiment-workflow`: 明确删除冗余配置或入口后，训练、评估、诊断工作流仍通过完整 resolved/final config 与现有 CLI 保持可复现。

## Impact

- 受影响源码区域包括 `src/kd_sensing/preprocessing/raymobtime_s008.py`、`src/kd_sensing/diagnostics/complementarity.py`、`src/kd_sensing/models/csi.py`、`tools/visualization/viewer_utils.py`、`tools/visualization/gradio_multimodal_viewer.py`、`src/kd_sensing/config/canonical.py`、`src/kd_sensing/config/canonical_recipes/`、`scripts/`、`tools/analysis/` 和相关测试。
- 需要更新 `tests/test_architecture_boundaries.py`、配置加载等价测试、CLI help smoke 和相关模块单元测试。
- 用户可见训练、评估、预处理、manifest 导出入口保持兼容；如删除重复 wrapper，必须保留对应 console script 或包内 CLI。
- 不新增第三方依赖，不改变默认数据目录，不删除或迁移任何本地数据和实验产物。
