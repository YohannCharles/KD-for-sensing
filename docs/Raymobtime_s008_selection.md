# Raymobtime s008 current snapshot beam selection

Raymobtime s008 在本仓库中定义为 current snapshot beam selection：给定当前 `coord`、`image`、`lidar` 和可选 `ray` 特征，预测当前最优 beam class，并同时预测当前 LOS/NLOS 与 link quality。

## 数据目录

默认数据根目录为：

```text
dataset/Raymobtime/s008
```

需要的本地文件布局：

```text
dataset/Raymobtime/s008/
  baseline_data/
    beam_output/
    coord_input/
    lidar_input/
    image_v2_input/
  raw_data/
    CoordVehiclesRxPerScene_s008.csv
    ray_tracing_data_s008_carrier60GHz.zip
```

也可以通过配置中的 `data_root` 指向外部目录；系统不会移动、复制或删除真实数据。`dataset/`、`outputs/`、cache、日志和 checkpoint 都是本地产物，默认不提交。

## 预处理

所有命令使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_audit.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_index.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_ray_features.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_cache.yaml
```

默认 cache 输出到 `outputs/raymobtime_s008/cache`。cache 包含 split index、beam/LOS/link labels、no-LOS ray 输入特征、with-LOS 审计特征、normalization metadata 和 split fingerprint。

ray-tracing zip 中的官方 `.hdf5` 条目会解析为 path-level ray rows，并生成不含 LOS flag 的模型输入 `ray_features_no_los`、仅用于审计的 `ray_features_with_los` 和独立的 `link_quality` target。预处理会在 ray path 全部缺失、link target 全 fallback，或训练 split 的 link target 标准差为 0 时拒绝生成可训练 cache；旧的全 `-120 dBm` cache 需要重新运行预处理生成，历史 TensorBoard event 文件不会自动重写。

Raymobtime image 模态复用现有 `resnet18_imagenet_rgb` encoder，dataset 会按 `rgb_imagenet` 契约提供 `[1, 3, 224, 224]` 输入。LiDAR 模态按 s008 的 3D occupancy grid 处理，输入为 `[1, C, D, H, W]`，模型使用 `raymobtime_lidar_3d_cnn`：3D Conv Stem -> 3D Residual Blocks -> Channel Attention -> Global AvgPool + Global MaxPool -> MLP Projection Head。

## 训练与评估

canonical 多任务配置：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/raymobtime/s008_multitask_selection.yaml
```

最小 smoke 配置：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/raymobtime/s008_smoke_selection.yaml
```

评估使用统一入口，并显式传入 checkpoint：

```bash
conda run -n kd_mm_beam kd-sensing-evaluate \
  --config configs/raymobtime/s008_multitask_selection.yaml \
  --weights outputs/raymobtime_s008/experiments/s008_multitask_selection/checkpoints/best.pth
```

指标包括 `beam_top1/top3/top5`、当前 beam 距离敏感指标 `beam_dba_current` / `val_beam_dba`、`los_accuracy/f1/auc`、`link_mae/rmse/r2` 和 `selection_multitask_loss`。Raymobtime current snapshot objective 不写 legacy future-only `val_adba`。当某个 split 的 LOS 只有单类时，AUC 会记录为不可用状态。

单任务 objective：

- `current_beam_selection`：当前最优 beam class 分类，默认早停 `val_beam_top1/max`
- `current_los_classification`：当前 LOS/NLOS 分类，默认早停 `val_los_f1/max`
- `current_link_quality`：当前 link quality 回归，默认早停 `val_link_mae/min`

## 实验边界

- sensing-only：`coord+image+lidar`
- sensing+ray：任何包含 `ray` 的组合，例如 `coord+image+lidar+ray`

推荐实验矩阵包含 `coord`、`image`、`lidar`、`ray` 单模态，以及 `coord+image`、`coord+lidar`、`coord+ray`、`image+lidar`、`coord+image+lidar` 和 `coord+image+lidar+ray`。每组可选择 `simple_concat_multitask_selection` 或 `task_aware_gated_multitask_selection`。

sensing-only 单任务主矩阵为 12 个 run：`coord`、`image`、`lidar`、`coord+image+lidar` × `current_beam_selection`、`current_los_classification`、`current_link_quality`。包含 `ray` 的 run 作为 sensing+ray 补充实验单独标注。

本仓库不再提供专用模态失衡分析入口。查看 Raymobtime s008 结果时使用常规 `metrics.json`、`test_report.json`、训练日志和统一评估入口；如需额外分析，应在新的 OpenSpec change 中定义输入、输出和验收标准。
