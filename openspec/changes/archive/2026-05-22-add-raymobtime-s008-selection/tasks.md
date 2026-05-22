## 1. 架构与配置基础

- [x] 1.1 在 `src/kd_sensing/data/layouts.py` 增加 Raymobtime family descriptor，默认解析 `dataset/Raymobtime/s008`，并保留显式 `data_root` 覆盖。
- [x] 1.2 新增 `configs/preprocess/raymobtime_s008_audit.yaml`、index/ray/cache 构建配置和 `configs/raymobtime/s008_multitask_selection.yaml` 初版配置。
- [x] 1.3 更新 `src/kd_sensing/registries.py` 的默认组件导入，确保 Raymobtime dataset、preprocessors、models 和 metrics 只在构建组件时注册。
- [x] 1.4 添加 Raymobtime s008 小 fixture，用于单测覆盖 CSV、beam npz 三类格式和简化 ray feature 输入，不提交真实数据或大 cache。

## 2. Raymobtime 预处理与 cache

- [x] 2.1 在 `src/kd_sensing/preprocessing/raymobtime_s008.py` 实现审计 preprocessor，输出必需路径检查、npz keys/shape/dtype、CSV 分布和坐标范围摘要。
- [x] 2.2 实现 snapshot index builder，只保留 `Val == "V"`，生成稳定 `sample_id`、split index CSV 和 split metadata。
- [x] 2.3 实现 beam 标签标准化，支持 `[N]`、`[N, 2]`、`[N, Tx, Rx]`，输出 `beam_label`、`beam_tx`、`beam_rx` 和 beam 维度 metadata。
- [x] 2.4 实现 ray-tracing feature extractor，生成 no-LOS 模型输入特征、with-LOS 审计特征和 link quality target，并记录 unmatched report。
- [x] 2.5 实现 cache builder，将 coord/image/lidar/ray 输入索引、labels、targets、normalization metadata 和 split fingerprint 写入配置指定 ignored 目录。
- [x] 2.6 扩展 `kd-sensing-preprocess` action 或配置 type 解析，让 Raymobtime 预处理可通过 `conda run -n kd_mm_beam kd-sensing-preprocess --config ...` 运行。
- [x] 2.7 使用 `conda run -n kd_mm_beam pytest tests/... -q` 添加并运行 Raymobtime 预处理/cache fixture 测试。

## 3. Dataset 与模态契约

- [x] 3.1 扩展 `src/kd_sensing/modalities.py`，新增 `coord` 和 `ray` 的 sample keys、fusion input keys、dataset flags、默认输入维度和标准化规则。
- [x] 3.2 在 runtime batch helper 中实现 `prepare_coord_inputs`、`prepare_ray_inputs`，并让 fusion 输入准备和 `forward_model` 支持 `coord_batch`、`ray_batch`。
- [x] 3.3 实现并注册 `RaymobtimeS008SnapshotDataset`，返回 flat snapshot batch 字段和 metadata，不返回 history/future/horizon 字段。
- [x] 3.4 在 data factory、run metadata 和 normalization artifact 路径中接入 Raymobtime dataset 的 split metadata、样本数、beam 维度、LOS/link target metadata。
- [x] 3.5 添加 dataset contract 单测，使用 `conda run -n kd_mm_beam pytest tests/... -q` 验证字段、shape、错误提示和旧配置不启用 `coord/ray`。

## 4. Prediction Objective、Loss 与 Metrics

- [x] 4.1 扩展 `src/kd_sensing/engine/prediction_objectives.py`，新增 `current_beam_selection` 和 `selection_multitask` objective metadata、early stopping alias 和 runtime metadata。
- [x] 4.2 实现 Raymobtime target 准备逻辑，读取 `target_beam`、`los_label`、`link_quality` 并对齐当前 snapshot 单步输出。
- [x] 4.3 实现 selection multitask loss，按配置权重合成 beam CE、LOS BCEWithLogits 和 link SmoothL1，并记录分项 loss。
- [x] 4.4 扩展验证/评估流程，输出 `beam_top1/top3/top5`、`los_accuracy/f1/auc`、`link_mae/rmse/r2` 和 `val_selection_multitask_loss`。
- [x] 4.5 处理 LOS AUC 单类 split 的 unavailable/null 状态，避免静默返回错误数值。
- [x] 4.6 使用 `conda run -n kd_mm_beam pytest tests/... -q` 添加并运行 objective、loss、metrics 和 early stopping 校验测试。

## 5. Snapshot 多任务模型

- [x] 5.1 实现 `simple_concat_multitask_selection` 模型注册，支持 `coord/image/lidar/ray` 任意非空组合和 beam/LOS/link 三类输出。
- [x] 5.2 实现 `task_aware_gated_multitask_selection` 模型注册，按 `beam_selection`、`los`、`link_quality` 输出 task-specific gates 和 diagnostics。
- [x] 5.3 为 coord/ray 添加快照向量 encoder/projector；Raymobtime image 复用现有 `resnet18_imagenet_rgb`，LiDAR 使用 3D occupancy grid 专用 `raymobtime_lidar_3d_cnn`。
- [x] 5.4 添加模型输入维度校验，确保 Raymobtime snapshot 模型收到时间维大于 1 时抛出清晰错误，且模块树不包含 GRU/RNN/LSTM。
- [x] 5.5 使用 `conda run -n kd_mm_beam pytest tests/... -q` 添加并运行模型 forward、diagnostics 和 no-temporal-core 测试。
- [x] 5.6 移除 Raymobtime image/lidar 轻量 2D encoder 路径，改为 registry encoder 配置：image 走共享 ResNet-18，lidar 走轻量 3D CNN + 全局池化 + 投影头。

## 6. Workflow、分析与文档

- [x] 6.1 新增 Raymobtime s008 smoke 配置或测试入口，覆盖 dataset -> dataloader -> model -> loss -> validation -> checkpoint 的最小路径。
- [x] 6.2 实现 Raymobtime 模态失衡分析 CLI 或包内分析入口，输出单模态性能、gate 均值、modality drop delta、LOS bucket 指标和贡献诊断。
- [x] 6.3 在 README 或专门文档中记录 Raymobtime s008 数据目录、预处理命令、训练/评估命令、sensing-only 与 sensing+ray 结果边界。
- [x] 6.4 确保文档和配置使用 current snapshot beam selection 表述，不使用 future beam prediction、beam tracking、history window 或 future horizon 表述。
- [x] 6.5 使用 `conda run -n kd_mm_beam kd-sensing-preprocess --help`、`conda run -n kd_mm_beam kd-sensing-train --help` 和 Raymobtime smoke 命令验证 CLI 可用性。

## 7. 验证与收尾

- [x] 7.1 运行 `openspec validate add-raymobtime-s008-selection --strict` 并修复所有 OpenSpec 问题。
- [x] 7.2 运行 `openspec status --change add-raymobtime-s008-selection` 确认 artifacts 和 tasks 状态。
- [x] 7.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证架构边界未回退。
- [x] 7.4 运行 Raymobtime 相关 focused tests：`conda run -n kd_mm_beam pytest tests/... -q`。
- [x] 7.5 在实现完成后运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归。

## 8. Raymobtime 单任务矩阵补充

- [x] 8.1 扩展 Raymobtime selection objective，支持 `current_los_classification` 与 `current_link_quality` 作为独立单任务训练目标。
- [x] 8.2 更新 Raymobtime 配置校验、指标可用性和 early stopping 默认值，使 LOS/link 单任务不依赖 future-only DBA。
- [x] 8.3 给出 sensing-only 12-run 主矩阵命令：`coord`、`image`、`lidar`、`coord+image+lidar` × `beam_selection`、`los`、`link_quality`。
