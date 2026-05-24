## Why

项目即将接入 Multimodal-NF 这类低空 UAV 近场 XL-MIMO 数据集。它使用 HDF5 frame-wise 样本、3D 位置、RGB、LiDAR 点云、近场 CSI、Top-5 三维码本 beam 和 LoS/NLoS/NF 标签，和当前 DeepSense6G 的 CSV 序列 future beam 预测不是同一种数据语义。

现有 dataset 代码已经出现不同数据集各自处理路径、样本索引、target 和 cache 的分叉；如果继续直接堆 `Dataset` 子类，后续 UAV、3D codebook、空间预测任务会继续放大这个问题。因此本变更把 Multimodal-NF 接入和 dataset runtime 抽象一起设计，但按阶段迁移，避免一次性重写现有工作流。

## What Changes

- 新增 `multimodal_nf` dataset type，默认读取 `dataset/MultimodalNF/` 下的 Multimodal-NF 本地数据，并保留显式 `data_root` 覆盖。
- 新增 Multimodal-NF HDF5 审计、index/cache 构建、city-level split metadata 和 codebook metadata 解析流程；不下载、不移动、不提交真实数据。
- 新增 dataset runtime contract：`DatasetDescriptor`、`SampleIndex`、`ModalityAdapter`、`TargetProvider` 和薄 `RuntimeDataset` 组合方式，用于统一 DeepSense6G、MMW、Raymobtime 和 Multimodal-NF 的接入边界。
- 新增 Multimodal-NF frame-wise dataset 契约：按启用模态返回 `image`、`lidar`、`gps`、`csi` 等 flat batch 字段，返回当前近场 beam target、Top-5 beam triplet/power、LoS/NLoS、NF/FF 和 metadata。
- 扩展模态契约，使既有 `gps` 和 `csi` 可以声明 dataset-specific profile：Multimodal-NF 的 `gps` 表示当前 UAV 3D 位置，`csi` 表示近场 XL-MIMO channel tensor，而不复用 DeepSense6G 的历史 GPS relative-polar 或 CSI 降级语义。
- 扩展 data factory，使非 CSV 或 HDF5/cache-backed dataset 不再被强制套用 DeepSense6G split CSV 规则。
- 新增近场 beam selection 目标与 3D codebook target schema，支持把 `BeamIdx[N, 5, 3]` 映射到 flattened class，同时保留三维 triplet、Top-5 候选和 beam power 用于指标与诊断。
- 提供 Multimodal-NF 迁移路线：先做 descriptor/index/contract，再接入 dataset 和目标，最后逐步把 DeepSense6G/MMW 迁移到同一 runtime contract。
- 不新增绕过 `src/kd_sensing` 包结构的旧入口脚本；所有预处理、训练、评估和测试继续走包内 CLI 或 `scripts/` 现有入口。

## Capabilities

### New Capabilities

- `dataset-runtime-contracts`: 定义跨数据集家族复用的 descriptor、sample index、modality adapter、target provider、metadata 和 artifact 传递契约。
- `multimodal-nf-dataset`: 定义 Multimodal-NF 数据布局、HDF5 审计、dataset 输出字段、near-field 3D codebook target、配置和验证工作流。

### Modified Capabilities

- `dataset-directory-layout`: 增加 Multimodal-NF 数据集家族目录、HDF5/codebook/cache/输出边界。
- `modality-contracts`: 扩展模态 profile 契约，使 `gps` 与 `csi` 能表达 Multimodal-NF 当前 UAV 3D 位置和近场 XL-MIMO channel 输入。
- `modality-aware-data-loading`: 扩展 dataset 构建流程，支持 descriptor-driven 非 CSV 数据集和 HDF5/cache-backed split 解析。
- `first-class-prediction-tasks`: 增加近场 beam selection 目标、3D codebook target schema、Top-K/Top-5 相关指标和运行 metadata。

## Impact

- 受影响源码：`src/kd_sensing/data/layouts.py`、`src/kd_sensing/data/datasets/`、`src/kd_sensing/data/transform_ops/`、`src/kd_sensing/modalities.py`、`src/kd_sensing/engine/data_factory.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/prediction_objectives.py`、`src/kd_sensing/evaluation/`、`src/kd_sensing/preprocessing/`、`src/kd_sensing/registries.py`。
- 受影响配置：新增 Multimodal-NF 预处理、dataset smoke、单模态和 fusion near-field beam selection 配置；旧 DeepSense6G/MMW/Raymobtime 配置必须保持兼容。
- 依赖与数据格式：需要 HDF5 读取支持；真实数据和 codebook 文件来自用户本地 `dataset/MultimodalNF/` 或显式路径，不作为源码变更提交。
- 验证要求：新增小型 fixture 覆盖 HDF5 schema、Top-5 3D codebook、dataset output、data factory 和 objective；相关 Python 命令必须使用 `conda run -n kd_mm_beam ...`。
