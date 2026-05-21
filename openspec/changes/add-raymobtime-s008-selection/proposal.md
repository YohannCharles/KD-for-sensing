## Why

Raymobtime s008 是单场景快照数据集，适合定义为 sensing-aided current beam selection，而不是当前项目默认的 DeepSense6G future beam prediction 或 snapshot next-frame baseline。项目需要一个和现有 `src/kd_sensing` 包结构、配置驱动训练入口、模态契约和 OpenSpec 架构一致的方案，用于研究 Raymobtime s008 上不同任务的模态偏好和多任务模态失衡。

## What Changes

- 新增 Raymobtime s008 数据集家族支持，默认数据根目录为 `dataset/Raymobtime/s008`，本地原始数据、cache、审计输出和训练输出仍不进入源码变更。
- 新增 s008 数据审计、snapshot index 构建、beam/LOS/link 标签标准化和 ray-tracing path-level 特征提取流程，输出可复现的轻量 cache 与 split metadata。
- 新增 `raymobtime_s008` dataset type，按 snapshot 样本返回当前 `coord`、`image`、`lidar`、`ray` 输入和当前 `beam_selection`、`los`、`link_quality` targets，不暴露 history/future/horizon 语义。
- 扩展模态契约以支持 `coord` 和 `ray` 两个非时序快照模态，并继续复用既有 `image`、`lidar` 模态。
- 新增 Raymobtime current beam selection / LOS / link quality 多任务目标，指标包括 beam Top-1/Top-3/Top-5、LOS accuracy/F1/AUC 和 link MAE/RMSE/R2。
- 新增 Raymobtime LOS 与 link quality 单任务 objective，支持 sensing-only 12-run 主矩阵：`coord`、`image`、`lidar`、`coord+image+lidar` × beam/LOS/link。
- 新增 snapshot 多模态模型配置和模型注册，包括简单拼接多任务模型与 task-aware gated 多任务模型；image 复用现有 `resnet18_imagenet_rgb`，Raymobtime LiDAR 使用 3D occupancy grid 专用轻量 3D CNN，gate diagnostics 必须按任务输出。
- 新增 Raymobtime s008 canonical 配置、smoke workflow 和模态失衡分析输出，区分 sensing-only 与 sensing+ray 实验。
- 明确禁止在 Raymobtime s008 第一版中引入 future beam prediction、beam tracking、历史窗口、future horizon、GRU/temporal transformer、LOS transition prediction 或 beam switch prediction。

## Capabilities

### New Capabilities

- `raymobtime-s008-selection`: 覆盖 Raymobtime s008 snapshot 数据准备、dataset 契约、current beam selection/LOS/link 多任务训练、模型输出、指标和模态失衡分析。

### Modified Capabilities

- `dataset-directory-layout`: 增加 Raymobtime 数据集家族和 s008 默认目录布局。
- `modality-contracts`: 增加 `coord` 和 `ray` 模态契约，并要求模态标准化、dataset flag、batch 输入和模型默认字段支持这两个模态。
- `first-class-prediction-tasks`: 增加 Raymobtime snapshot current beam selection 多任务目标及其 target、loss、metric、early stopping 和运行产物契约。
- `experiment-workflow`: 增加 Raymobtime s008 的配置驱动预处理、训练、评估、smoke 和分析 workflow。

## Impact

- 受影响源码：`src/kd_sensing/data/layouts.py`、`src/kd_sensing/modalities.py`、`src/kd_sensing/data/datasets/`、`src/kd_sensing/preprocessing/`、`src/kd_sensing/engine/`、`src/kd_sensing/evaluation/`、`src/kd_sensing/models/`、`src/kd_sensing/cli/` 和 `src/kd_sensing/registries.py`。
- 受影响配置：新增 `configs/raymobtime/s008_multitask_selection.yaml` 及必要的模型/分析配置；不新增绕过包结构的旧入口脚本。
- 受影响命令：所有 Python 预处理、训练、评估和测试命令必须通过 `conda run -n kd_mm_beam ...` 执行。
- 依赖风险：ray-tracing zip 解析格式需要先通过审计工具确认；首版实现必须容忍缺失真实数据，并通过 synthetic/small fixture 覆盖核心契约。
- 输出边界：`dataset/`、`outputs/`、cache、日志、checkpoint 和审计产物仍为本地产物，不纳入源码变更。
