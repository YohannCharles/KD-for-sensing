## Context

当前项目的主工作流围绕 DeepSense6G 序列数据展开，训练入口通过 `scripts/train.py`、`kd-sensing-train`、registry、中心化模态契约和 `src/kd_sensing` 包内模块组织。已有 snapshot next-frame baseline 仍是“当前帧预测下一帧”的 future target 变体，不等同于 Raymobtime s008 的单场景 snapshot beam selection。

`Raymobtime数据集.md` 明确要求 s008 第一版定义为“当前多模态快照 -> 当前最优 beam pair”，并同时预测当前 LOS/NLOS 与 link quality。该文档中给出的顶层脚本和 `src/data`/`src/models` 目录写法需要调整为本仓库现有架构：新增能力必须走 `src/kd_sensing`、配置驱动入口、registry 和 OpenSpec 契约，不能新增绕过包结构的旧入口。

## Goals / Non-Goals

**Goals:**

- 支持 Raymobtime s008 作为独立 dataset family，默认目录为 `dataset/Raymobtime/s008`，同时允许用户显式传入外部 `data_root`。
- 提供可单独运行的数据审计、index/cache 构建、ray-tracing 特征提取和 split metadata 生成流程。
- 提供 `raymobtime_s008` dataset type，使用 snapshot 样本契约接入现有 DataLoader、runtime batch helper 和训练/评估输出体系。
- 在中心化模态契约中新增 `coord` 与 `ray`，并让 `coord/image/lidar/ray` 任意非空组合可被 Raymobtime 模型和分析流程使用。
- 增加 current beam selection 与 beam/LOS/link 多任务目标，支撑 task-dependent modality imbalance 分析。
- 提供 simple concat 和 task-aware gated 两类无时序 snapshot 模型，gate diagnostics 能按任务和模态导出。
- 所有 Python 命令、测试和 smoke 运行均使用 `conda run -n kd_mm_beam ...`。

**Non-Goals:**

- 不实现 future beam prediction、beam tracking、beam switch prediction、LOS transition prediction 或任何跨 episode/scene 的时序预测。
- 不在首版引入 GRU/RNN/LSTM/temporal transformer，也不复用 DeepSense6G 历史窗口 CSV 语义。
- 不下载、移动、复制或提交 Raymobtime 真实数据、cache、日志、checkpoint 或审计输出。
- 不把 ray-tracing 派生特征作为 sensing-only 结果的一部分；包含 `ray` 的实验必须单独标注为 sensing+ray。
- 不泛化到全部 Raymobtime/CAVIAR 场景；本 change 只覆盖 s008 snapshot selection。

## Decisions

### 1. 使用包内预处理与 CLI，而不是新增顶层脚本

Raymobtime 文档中的 `scripts/audit_s008_files.py`、`scripts/build_s008_index.py` 等入口会改为 `kd_sensing.preprocessing.raymobtime_s008` 中的 PREPROCESSORS，并通过 `kd-sensing-preprocess --config ... --action ...` 或等价包内 CLI 调用。这样可以复用现有配置加载、路径解析和 registry 机制，也符合项目“不新增旧入口”的约束。

替代方案是按文档新增多个顶层脚本；这会绕开现有 CLI、配置覆盖和路径解析体系，后续也更难纳入架构边界测试，因此不采用。

### 2. Raymobtime 使用 flat batch 契约，保留 snapshot 语义

用户文档给出的嵌套返回结构适合概念描述，但现有 trainer/runtime 依赖 flat dict batch。`RaymobtimeS008SnapshotDataset` 将返回 flat keys：`coord`、`image`、`lidar`、`ray`、`target_beam`、`los_label`、`link_quality` 和 `meta`。其中输入张量仍保留单步维度，例如 `coord: [1, F]`、`image: [1, 3, H, W]`、`lidar: [1, C, H, W]`、`ray: [1, F]`，`target_beam: [1]` 表示当前最优 beam class。

新 Raymobtime 代码和配置文档使用 `snapshot`、`current`、`beam_selection` 命名；共享 runtime 内部为兼容既有 helper 可能继续传递单步长度字段，但不得把它暴露为 future horizon 语义。

### 3. 新增 `coord` 和 `ray` 作为一等模态

`coord` 不复用现有 `gps`，因为 Raymobtime s008 的 coordinate input 是当前场景快照坐标，不是 DeepSense6G 的历史 GPS relative-polar 序列。`ray` 也不复用 `mmwave` 或 `csi`，因为它来自 ray-tracing path-level 派生特征，和传感器感知模态的实验解释不同。

新增模态会进入中心化 `MODALITY_ORDER`，但 canonical DeepSense6G 配置不默认启用它们。这样 Raymobtime 的模态选择、batch key、dataset flag、model default 和诊断输出能走同一套契约。

### 4. 新增 selection objective，而不是改写既有 `beam`

现有 `beam` objective 绑定 future beam 标签、DBA 和序列预测历史。Raymobtime s008 应使用新的 `current_beam_selection`、`current_los_classification`、`current_link_quality` 单任务 objective，以及 `selection_multitask` 多任务 objective。`selection_multitask` 包含 beam selection CE、LOS BCE 和 link SmoothL1 分量，默认 early stopping 使用 `val_selection_multitask_loss/min`，用户可以显式覆盖为 `val_beam_top1/max`。

替代方案是把 s008 映射到现有 `multitask`；这会把 LOS/link 误解为 occlusion/position，也会污染既有 DeepSense6G 多任务契约，因此不采用。

### 5. ray-tracing 特征分成 no-LOS 输入与 with-LOS 审计版本

ray 特征 cache 必须至少生成 `ray_features_no_los` 和 `ray_features_with_los`。模型输入只能使用 no-LOS 版本，防止 LOS 分类标签泄漏；with-LOS 版本只用于审计、对齐验证和数据质量报告。link quality label 可来自 ray-tracing received power，但必须作为 target 单独保存。

### 6. 模型第一版限制为无时序 snapshot 多任务模型

首版只注册 `simple_concat_multitask_selection` 与 `task_aware_gated_multitask_selection`。两者通过 registry 构建各模态 encoder，输出 `[B, 1, C]` beam logits、`[B, 1]` LOS logits 和 `[B, 1]` link prediction。Task-aware gate 为每个任务输出 `[B, K]` gate，并在 diagnostics 中携带模态名，供后续分析。

Raymobtime 不再为 image/lidar 另建一套 2D 轻量 encoder。`image` 模态 MUST 复用项目现有 `resnet18_imagenet_rgb` encoder 和 `rgb_imagenet` 输入契约。`lidar` 模态由于 s008 baseline 输入是 3D occupancy grid，MUST 使用专用 `raymobtime_lidar_3d_cnn` encoder：3D Conv Stem -> 3D Residual Blocks -> Channel Attention -> Global AvgPool + Global MaxPool -> MLP Projection Head -> LiDAR embedding。`coord` 与 `ray` 仍使用 registry 中的快照向量 MLP encoder。

Raymobtime image cache 可能来自低分辨率 `image_v2_input`，训练入口应避免在 dataset `__getitem__` 内逐样本 CPU resize。默认 Raymobtime 主配置将 image resize 延迟到 runtime batch 准备阶段，在 batch tensor 搬到目标 device 后统一 resize 到 `rgb_imagenet` 的 224x224 输入尺寸，并限制 PyTorch CPU intra/inter-op 线程数以支持多个单模态 run 并行。

### 7. 分析输出必须区分 sensing-only 与 sensing+ray

Ray-tracing 特征和通信标签源距离很近，可能形成强模态。实验矩阵和汇总报告必须把 `coord+image+lidar` 与 `coord+image+lidar+ray` 分开标注，并提供 test-time modality drop、gate mean、按 LOS bucket 的指标和梯度/贡献诊断。

## Risks / Trade-offs

- [Risk] Raymobtime s008 的 npz/CSV/zip 字段格式可能和文档假设不完全一致 → 先实现审计工具和小 fixture，根据审计结果让 cache builder 支持多种 beam 输出形状，并在错误信息中打印实际 keys/shape。
- [Risk] 新增 `coord`/`ray` 扩展中心化模态契约，可能影响 canonical fusion 解析 → 默认 canonical 配置不包含这两个模态；新增架构边界测试验证旧模态顺序和旧配置仍可加载。
- [Risk] LOS AUC 在小 split 或单类 split 下不可定义 → metrics 返回结构化 unavailable/null 状态，同时 accuracy/F1 继续输出。
- [Risk] link quality label 的单位和聚合口径可能需要依据真实 zip 校验 → cache metadata 必须记录 `link_target_name`、单位、聚合方式和特征提取版本。
- [Risk] 复用现有 runtime 需要适配单步 snapshot 张量 → Raymobtime 模型必须拒绝时间维不等于 1 的输入，并用 smoke test 覆盖 train/eval forward、loss、metrics 和 checkpoint 路径。

## Migration Plan

1. 添加 OpenSpec 契约、配置样例和小 fixture，不触碰真实数据。
2. 实现 layout descriptor、模态契约和 PREPROCESSORS，并用 fixture 跑审计/cache 单测。
3. 实现 dataset、batch/runtime 适配、objective、metrics 和模型注册。
4. 添加 `configs/raymobtime/s008_multitask_selection.yaml` 与 smoke 测试，使用 `conda run -n kd_mm_beam ...` 验证。
5. 添加分析 CLI 与文档说明，确保本地产物仍落在 `outputs/` 或配置指定的 ignored 路径。

Rollback 策略：该 change 主要新增 dataset type、模态和 objective；若真实数据格式阻塞，可保留不默认启用的代码路径并回退 Raymobtime 配置入口，不影响现有 DeepSense6G/MMW/CSI 工作流。

## Open Questions

- `baseline_data/beam_output` 在本地真实数据中的 npz key 和 shape 需要通过审计确认，首版 cache builder 应先支持 `[N]`、`[N, 2]` 和 `[N, Tx, Rx]` 三类。
- ray-tracing zip 内 path-level 文件命名与 sample 对齐键需要用真实 s008 数据确认；如果无法从 zip 直接按 `EpisodeID/SceneID/VehicleArrayID` 定位，需要先输出 unmatched report。
- link quality 默认 target 采用 `link_power_max_dbm` 还是 `link_power_sum_dbm` 需要由首轮审计后的标签分布决定；配置必须允许显式选择。
