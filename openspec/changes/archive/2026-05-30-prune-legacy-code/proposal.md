## Why

MARF、CRAF、G2D 和 Multimodal-NF 相关研究线已经不再作为当前项目目标维护，但它们仍占据模型注册、训练分支、配置 overlay、数据集契约、诊断日志和测试面，持续拉高维护成本并干扰后续核心工作。现在应一次性删除这些旧代码和规格承诺，不保留兼容入口，让仓库聚焦当前仍需要的标准单模态、通用 fusion、MMW/DeepSense/Raymobtime 等有效工作流。

## What Changes

- **BREAKING**：删除 MARF 架构，包括 `marf_fusion` 模型、MARF router/prior/subset training 逻辑、MARF 配置 overlay、MARF 训练 helper、日志字段和测试。
- **BREAKING**：删除 CRAF 架构，包括 `craf_fusion`、teacher-prior CRAF、counterfactual/reliability gate、CRAF loss helper、teacher registry/encoder loader 中仅服务 CRAF/MARF 的逻辑、配置、日志和测试。
- **BREAKING**：删除 G2D 多模态蒸馏，包括 `distillation.type: g2d`、G2D distiller/SMP/diagnostics、G2D teacher ensemble 构建、G2D 配置 alias/overlay、G2D 日志文件和测试。
- **BREAKING**：删除 Multimodal-NF 数据集家族，包括 `data.dataset.type: multimodal_nf`、相关 dataset/preprocessing/runtime/cache/profile 代码、`configs/multimodal_nf/` 和 `configs/preprocess/multimodal_nf_*.yaml`、数据集规格、运行产物语义和测试 fixture。
- 删除 README、docs、OpenSpec 中对 MARF/CRAF/G2D/Multimodal-NF 的推荐入口、实验矩阵、健康检查和数据布局说明。
- 清理相关本地日志/输出引用和生成路径约束；源码不主动删除用户本地真实数据或历史 `outputs/`，但仓库不再提供读取、生成或测试这些旧产物的入口。
- 删除或改写相关测试脚本，最终保留只覆盖当前仍支持能力的架构边界、配置加载、CLI help、数据集和训练 smoke 检查。
- 不提供 deprecated alias、兼容 facade、自动迁移脚本或旧配置重定向；外部旧配置应直接失败并提示未知注册名、未知 dataset type 或未知 distillation type。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 明确旧研究线删除后的源码边界、产物边界和健康检查范围，不再要求保留 MARF/CRAF/G2D/Multimodal-NF 模块。
- `component-registry`: 删除 CRAF/MARF/G2D/Multimodal-NF 相关模型、loss、distiller、dataset 和 preprocessor 注册承诺。
- `configurable-multimodal-fusion`: 删除 CRAF、MARF、G2D 高级 fusion 配置、overlay/alias 和输出特征适配要求。
- `experiment-workflow`: 删除 CRAF/MARF/G2D 训练、评估、日志、smoke test、验证命令和 Multimodal-NF 运行产物一致性要求。
- `teacher-prior-gated-craf`: 退役 teacher-prior CRAF/MARF capability，删除 teacher reliability registry、prior gate、encoder loading、stage workflow 和 diagnostics 要求。
- `g2d-multimodal-distillation`: 退役 G2D capability，删除 distiller、teacher ensemble、SMP、confidence ranking 和 diagnostics 要求。
- `multimodal-nf-dataset`: 退役 Multimodal-NF dataset capability，删除本地布局、审计、index、sample、target、cache、runtime 和 smoke workflow 要求。
- `dataset-directory-layout`: 删除 Multimodal-NF 数据集家族目录规范和本地产物边界要求。
- `dataset-runtime-contracts`: 删除 Multimodal-NF runtime contract、objective metadata 和缓存运行语义要求。
- `first-class-prediction-tasks`: 删除仅服务 Multimodal-NF objective/head/schema 的任务契约。
- `modality-aware-data-loading`: 删除 Multimodal-NF dataset/data factory/cache/profile 相关数据加载要求。
- `modality-contracts`: 删除 Multimodal-NF 特定 image/LiDAR/GPS/CSI profile 与 batch contract 要求。
- `training-throughput-optimization`: 删除 Multimodal-NF cache/throughput/profile 相关优化要求。

## Impact

- 影响代码：`src/kd_sensing/models/fusion/craf.py`、`src/kd_sensing/models/fusion/marf.py`、`src/kd_sensing/engine/craf_training.py`、`src/kd_sensing/engine/marf_training.py`、`src/kd_sensing/engine/g2d_training.py`、`src/kd_sensing/engine/multimodal_nf_runtime.py`、`src/kd_sensing/distillation/{craf_losses.py,g2d.py,g2d_smp.py}`、`src/kd_sensing/diagnostics/g2d_diagnostics.py`、`src/kd_sensing/data/datasets/multimodal_nf.py`、`src/kd_sensing/preprocessing/multimodal_nf*.py` 以及它们的 import/registry/config glue。
- 影响配置：删除 `configs/multimodal_nf/`、`configs/preprocess/multimodal_nf_*.yaml`、CRAF/MARF/G2D fusion overlay 或实体配置、CSI hardening 中的 G2D-style 分支，以及 README/docs 中对应命令。
- 影响测试：删除 `tests/test_craf_*`、`tests/test_marf_*`、`tests/test_g2d_*`、`tests/test_multimodal_nf_*` 等正向测试，补充退役入口失败测试和当前保留能力 smoke。
- 影响文档和 OpenSpec：删除或修改所有仍要求这些旧研究线存在的规格和文档段落。
- 兼容性：外部旧配置、旧 checkpoint 分析脚本和历史日志解析脚本不再受支持；历史本地产物可作为静态文件留在用户磁盘，但项目不再保证能消费它们。
