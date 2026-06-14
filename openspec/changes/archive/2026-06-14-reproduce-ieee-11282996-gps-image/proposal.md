## Why

当前仓库已有 DeepSense6G Image+GPS、BeamBench/Arnold22 和 GPS+LiDAR BGAM 相关能力，但还没有针对 IEEE Xplore article `11282996` 的 AMR-Net-gps-image 可审计复现入口。用户明确要求只使用 GPS 和 image 模态、不使用 LiDAR，因此需要单独收口论文协议、数据场景、模型组、指标和 claim 边界，避免误复用现有 GPS+LiDAR 或 all-modalities workflow。

Source audit 发现需要额外收窄 claim：公开 Crossref metadata 将 IEEE document `11282996` 指向 DOI `10.1109/JIOT.2025.3641184` 的 IEEE Internet of Things Journal AMR-Net 论文；而 DeepSense6G Scenario 23 的公开作者页面/代码包对应的是 GLOBECOM 2022 论文 `Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction`，DOI `10.1109/GLOBECOM48099.2022.10000718`，IEEE document `10000718`。因此本 change 实现的 Scenario 23 GPS+Image workflow 只能作为 local substitute / mock smoke / paper-protocol-audited helper，默认必须把 IEEE `11282996` official reproduction 标记为 `blocked_official` 或 metadata conflict，直到用户提供可核对的 IEEE PDF/BibTeX/官方协议证据。

## What Changes

- 新增 AMR-Net-gps-image 复现能力：先审计 IEEE `11282996` 页面/PDF、官方或作者代码、数据场景、split、模态、label space、指标和目标表格，再生成本仓库可运行的 local reproduction workflow；若 audit 发现 article metadata 与 Scenario 23 作者包不一致，report MUST 明确记录 conflict 并禁止 official claim。
- 新增 paper-specific manifest/config preset，强制 `model.modalities` 与 dataset 启用模态只包含 `image` 和 `gps`；`lidar`、`radar`、`mmwave`、`csi` 均不得被该复现入口读取或作为 fallback 输入。
- 支持论文所需 DeepSense6G 场景描述符和默认路径；公开作者页面/代码指向 drone beam prediction 的 Scenario 23，implementation 必须在 source audit 中确认后固化为场景描述符、CSV 名和输出分区。
- 复用 Vision-Position baseline suite，新增 paper `11282996` 模型组：image-only、GPS-only 和 Image+GPS fusion 的论文对齐配置；若论文只报告部分模型组，未报告行必须标记为 local control 而不是论文复现行。
- 输出论文对齐指标与报告：至少覆盖 Top-1、Top-3、Top-5 beam accuracy、可用时的 DBA/beam-distance 指标、beam-training overhead reduction 或 paper 中等价字段，并记录 source audit digest、数据/split、checkpoint provenance 和运行命令。
- 明确 claim status：在未获得 IEEE PDF/官方代码/官方数据打包/官方权重或 exact training search 之前，结果只能标记为 `local substitute`、`blocked official`、`mock/smoke` 或 `paper-protocol-audited`，不得声明为 official reproduction。
- 不引入 breaking change；现有 BEAMBench、JEPA、Scenario D、GPS+LiDAR BGAM、MMW 和通用训练/评估入口继续兼容。

## Capabilities

### New Capabilities

- `ieee-11282996-gps-image-reproduction`: 定义 AMR-Net-gps-image 的 source audit、GPS+Image-only 复现协议、禁止 LiDAR 的输入边界、模型组、指标、报告和 claim status。

### Modified Capabilities

- `deepsense6g-scene-selection`: 增加论文复现所需 DeepSense6G 场景描述符、默认路径、CSV 名和 scene-scoped 输出隔离。
- `vision-position-baseline-suite`: 增加 AMR-Net-gps-image paper preset/model group、metadata 和 smoke/回归测试要求。

## Impact

- 代码影响：优先新增 `src/kd_sensing/baselines/<paper-or-deepsense>/` 下的 paper workflow helper、包内 CLI 或已登记薄 alias；复用现有 dataset、batch/runtime、Vision-Position 模型组件和 metrics helper，不复制通用训练循环。
- 配置影响：新增 paper-specific manifest/config，默认输出到 ignored 的 `outputs/analysis/ieee_11282996_gps_image/` 或 scene-scoped run 目录；配置必须显式禁用 LiDAR，并在加载时拒绝包含 LiDAR 的模态列表。
- 数据影响：新增或扩展 DeepSense6G 场景描述符，不改写真实 `dataset/`、split CSV、image/GPS 文件、beam target、checkpoint 或训练日志。
- 文档影响：更新 README/主线实验文档/结果账本时只记录命令、协议、claim status 和 caveat；不得提交真实 metrics、plots、checkpoint、cache 或 predictions。
- 测试影响：覆盖 source-audit schema、scenario descriptor、paper config loading、LiDAR 禁用、model forward smoke、metrics/report writer 和 CLI help；所有项目 Python 测试使用 `conda run -n kd_mm_beam ...`。
