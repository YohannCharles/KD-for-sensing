## Why

现有跨场景/跨天气 beam prediction 实验已经暴露出 source prior collapse、target support/outlier 和 split eligibility 混杂的问题，但还缺少一个可复现的 5% target-shot split 与 geometry-residual label 基础层来隔离“绝对 beam 标签分布漂移”本身。先把 split、几何粗 beam、残差标签和分布诊断做成可审计契约，可以在不引入新模型复杂度的前提下验证论文主假设，并为后续 residual predictor、target residual prior calibration 和 reliability-aware fusion 打底。

## What Changes

- 新增可复现的 source/target domain split 能力，支持按 scenario、weather/condition、scenario+weather、town+scenario+weather 定义 domain。
- 新增 5% target-shot 协议：target domain 拆分为 `target_labeled`、可选 `target_unlabeled` 和 `target_test`，并写出 split indices、采样 manifest、统计和防泄漏 metadata。
- 新增 geometry-based coarse beam 与 residual beam label 基础模块：从 BS/RSU-centric UE/CAV position 计算 azimuth、`beam_geo`、`beam_residual`、`residual_class` 和 `geo_sector`。
- 新增 source/target 分布诊断命令，输出 absolute beam、geometry beam、residual beam histogram，以及 KL/JS/Wasserstein/TV 等分布距离。
- 修改数据 runtime 和模态加载契约，使 dataset sample 可按配置暴露 residual label/geometry metadata，同时 target unlabeled batch 不暴露监督标签给训练 loss。
- 修改 MMW cross-scene adaptation protocol，使其显式支持 5% target-shot split 与 geometry-residual split 统计，且继续遵守 target_test 防泄漏边界。
- 不新增 neural network、AE/CL/MAE 预训练、residual predictor、target prior calibration 或 weather-aware fusion；这些作为后续变更建立在本基础能力之上。

## Capabilities

### New Capabilities

- `target-shot-domain-splitting`: 定义多场景/多天气 source-target domain、5% target_labeled 采样、target_unlabeled/test 隔离、split indices 持久化、统计和防泄漏 metadata。
- `geometry-residual-beam-labels`: 定义 geometry coarse beam、circular residual label、clipped residual class、dataset sample 字段和 residual-to-absolute beam 还原契约。
- `beam-distribution-shift-diagnostics`: 定义 source/target absolute、geometry、residual label 直方图和分布距离诊断产物。

### Modified Capabilities

- `dataset-runtime-contracts`: 增加 target-shot split runtime metadata、labeled/unlabeled target subset 防泄漏和 geometry-residual target schema 暴露要求。
- `modality-aware-data-loading`: 增加 residual/geometry label 字段的按需加载契约，确保未启用 geometry-residual label_space 时不改变既有 sample keys。
- `mmw-cross-scene-adaptation-protocol`: 增加 5% target-shot 与 geometry-residual split 统计在 MMW scenario/town/weather split 中的协议要求。

## Impact

- 受影响代码：dataset/sample index、target provider、split builder、MMW planner/manifest metadata、评估/诊断工具、配置解析和测试。
- 新增配置：`split.*`、`label_space.*` 和 distribution diagnostics 输出配置；所有新行为必须通过配置显式启用。
- 新增脚本或 CLI：用于生成/复用 split、打印 label/residual 分布统计和写出 JSON/CSV/PNG 诊断。
- 新增测试：split 无交集与可复现、target_labeled 约 5%、target_unlabeled 标签 guard、circular residual wrap-around、residual_to_beam 可逆、分布统计产物存在。
- 不引入新的外部深度学习依赖；若分布距离需要 SciPy，应使用仓库已有依赖或提供 NumPy fallback，并保持轻量导入边界。
