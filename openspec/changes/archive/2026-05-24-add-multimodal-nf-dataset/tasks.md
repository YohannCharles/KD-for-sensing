## 1. 外部 schema 与小型 fixture

- [x] 1.1 基于官方 Multimodal-NF HDF5/codebook 描述创建最小 fixture，覆盖 `H`、`Pos`、`BeamIdx`、`BeamPower`、`Has_LoS`、`Is_NF`、image 和 LiDAR 点云字段，不提交真实数据。
- [x] 1.2 为 fixture 写入 dense/small codebook metadata，覆盖三维 triplet flatten 和 Top-5 target 场景。
- [x] 1.3 添加 schema 审计测试，使用 `conda run -n kd_mm_beam pytest ... -q` 验证 fixture keys、shape、dtype 和缺失字段错误。

## 2. Dataset runtime contract

- [x] 2.1 新增 dataset descriptor 模块，注册 DeepSenseG、MMW、Raymobtime 和 Multimodal-NF 的 family、storage kind、默认路径、split 语义和支持 profile。
- [x] 2.2 新增 sample index 抽象和 row dataclass，支持 CSV sequence、NPZ snapshot 和 HDF5 frame 三类 storage kind。
- [x] 2.3 新增 modality adapter 接口，声明 sample key、profile、shape/dtype 校验、懒加载和 cache/normalization 能力。
- [x] 2.4 新增 target provider 接口，支持主 label、辅助 target、valid mask、target metadata 和 train-to-test artifact 复用。
- [x] 2.5 新增薄 RuntimeDataset 组合实现，并确保输出为 DataLoader 可默认 collate 的 flat dict。
- [x] 2.6 添加 runtime contract 单测，使用 `conda run -n kd_mm_beam pytest ... -q` 验证 descriptor 查询轻量、禁用模态不读取、metadata 输出和旧 dataset descriptor 兼容。

## 3. Multimodal-NF layout、审计与 index

- [x] 3.1 在 `src/kd_sensing/data/layouts.py` 增加 Multimodal-NF layout descriptor，默认根目录为 `dataset/MultimodalNF`，保留显式路径覆盖。
- [x] 3.2 实现 Multimodal-NF 审计 preprocessor，输出 HDF5 keys/shape/dtype、city 列表、样本数、codebook 路径、缺失项和数据 fingerprint。
- [x] 3.3 实现 HDF5 frame index builder，生成 `sample_id`、city/trajectory/frame 引用、资源引用和 split metadata。
- [x] 3.4 支持默认 city-level split，并允许配置显式 city 集合或 frame-level debug split。
- [x] 3.5 实现 codebook metadata 解析，记录 codebook shape、flatten order、num beam classes 和文件 fingerprint。
- [x] 3.6 添加预处理配置样例，并用 `conda run -n kd_mm_beam kd-sensing-preprocess --help` 与 fixture focused tests 验证入口。

## 4. 模态 profile、adapter 与 dataset

- [x] 4.1 扩展中心化模态契约，新增 `gps:uav_xyz_snapshot`、`csi:xl_mimo_nf`、`lidar:point_cloud_xyz_10000` profile 标准化和 metadata。
- [x] 4.2 扩展 batch 准备逻辑，使 runtime 能按 profile 校验 `[B,T,3]` GPS、`[B,T,M,K,2]` CSI 和 `[B,T,P,3]` LiDAR 点云。
- [x] 4.3 实现 Multimodal-NF channel/image/lidar/gps modality adapters，使用 worker-local HDF5 懒加载，不在 dataset 初始化时物化大数组。
- [x] 4.4 实现并注册 `multimodal_nf` dataset type，按启用模态返回 flat sample、target 字段和 metadata。
- [x] 4.5 添加 dataset contract 测试，使用 `conda run -n kd_mm_beam pytest ... -q` 验证单模态、fusion、禁用模态缺失文件、shape 错误和 metadata。

## 5. Near-field target、objective 与指标

- [x] 5.1 实现 Multimodal-NF target provider，将 `BeamIdx[N,5,3]` 转换为 `target_beam`，同时保留 `beam_triplet_topk` 和 `beam_power_topk`。
- [x] 5.2 扩展 prediction objective metadata，新增 `near_field_beam_selection`、默认主指标 `val_beam_top1` 和可用指标集合。
- [x] 5.3 实现 near-field target 准备和 loss 计算，默认使用 flattened beam classification，拒绝模型输出类别数与 codebook 不一致的情况。
- [x] 5.4 实现 near-field Top-1/Top-3/Top-5 指标，并确保该 objective 不产生 `val_adba`。
- [x] 5.5 在 run metadata、checkpoint metadata 和评估报告中记录 objective、codebook shape、flatten 规则、input profiles 和辅助标签可用性。
- [x] 5.6 添加 objective/loss/metrics 测试，使用 `conda run -n kd_mm_beam pytest ... -q` 验证 Top-K、metadata、错误提示和 early stopping alias。

## 6. 配置、模型接入与迁移护栏

- [x] 6.1 新增 Multimodal-NF audit、dataset smoke、CSI-only、GPS-only、image+LiDAR 和 fusion near-field beam selection 配置样例。
- [x] 6.2 为模型构建增加 profile 兼容校验：未声明支持点云 LiDAR 或 near-field CSI 的 encoder MUST 被清晰拒绝。
- [x] 6.3 增加 DeepSenseG、MMW、Raymobtime descriptor shim 或测试护栏，确保旧配置仍走原有 dataset 行为和输出字段。
- [x] 6.4 将 MMW 动态 CSV 补列、DeepSense sequence index 和 Raymobtime snapshot index 的后续迁移步骤写入设计备注或任务注释，暂不删除旧实现。
  - 迁移备注：本变更只新增 descriptor shim 与测试护栏。DeepSense `create_samples` 到 `SampleIndex`、MMW 动态 CSV 补列前移到 preparation/index builder、Raymobtime NPZ snapshot index 接入 runtime provider 的收敛工作保留为后续 OpenSpec change；旧 dataset 类和输出字段不在本变更中删除或改名。
- [x] 6.5 添加架构边界测试，确保新增 runtime contract 不成为旧 facade，不引入顶层旧入口脚本，不让轻量 descriptor 导入重依赖。

## 7. 文档与验证

- [x] 7.1 更新 README 或扩展文档，说明 Multimodal-NF 本地目录、数据下载位置、codebook 放置、预处理命令、训练命令和本地产物边界。
- [x] 7.2 运行 `openspec validate add-multimodal-nf-dataset --strict` 并修复所有 OpenSpec 问题。
- [x] 7.3 运行 `openspec status --change add-multimodal-nf-dataset` 确认 artifacts 和任务状态。
- [x] 7.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界。
- [x] 7.5 运行 Multimodal-NF focused tests：`conda run -n kd_mm_beam pytest tests/... -q`。
- [x] 7.6 在实现完成后视范围运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。

## 8. 运行期资源控制与 HDF5 取样性能

- [x] 8.1 为 Multimodal-NF 三模态 fusion 配置显式设置 `training.cpu_threads`，并使用较保守的 train/test DataLoader worker 覆盖，降低并行实验 CPU 过度并发。
- [x] 8.2 缓存 worker-local HDF5 dataset path/key 解析，避免旧 index 缺少 image/lidar key 时每个样本重复遍历 HDF5 文件结构。
- [x] 8.3 让 `scripts/profile_training_io.py` 复用训练线程配置，使 profile 与训练入口的运行期线程行为一致。
- [x] 8.4 为 Multimodal-NF 增加 split-specific `eval_portion` 验证抽样，并将连续 HDF5 history/target 读取改为 slice，降低常规调参时的验证和取样开销。
