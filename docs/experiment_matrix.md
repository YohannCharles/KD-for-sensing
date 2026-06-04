# 实验矩阵

本文件承载 README 中移出的实验矩阵和推荐运行顺序。命令默认使用 `kd_mm_beam` 环境；训练、评估和预处理优先使用 console script。

## 单模态和基础 Fusion

单模态 canonical 矩阵使用 strong、lightweight 和 supervised 三类入口。所有入口都构建单个 `model.primary` 主模型。

| 模态 | strong | lightweight | supervised |
| --- | --- | --- | --- |
| image | `configs/image/strong.yaml` | `configs/image/lightweight.yaml` | `configs/image/supervised.yaml` |
| radar | `configs/radar/strong.yaml` | `configs/radar/lightweight.yaml` | `configs/radar/supervised.yaml` |
| gps | `configs/gps/strong.yaml` | `configs/gps/lightweight.yaml` | `configs/gps/supervised.yaml` |
| lidar | `configs/lidar/strong.yaml` | `configs/lidar/lightweight.yaml` | `configs/lidar/supervised.yaml` |
| mmwave | `configs/mmwave/strong.yaml` | `configs/mmwave/lightweight.yaml` | `configs/mmwave/supervised.yaml` |

推荐主线顺序是先运行 supervised/adaptation baseline，再进入 DeepSense6G GPS residual、Top8 selector、GPS+LiDAR BGAM、MMW GPS v2/BGAM、Raymobtime s008、CSI hardening 或 viewer manifest。旧 `teacher_no_kd`、`student_no_kd`、`no_kd`、`logits_kd`、`rkd`、`configs/hist_beam/*` 和 HiST-Beam 入口不再作为支持入口存在；配置加载器会拒绝这些路径并给出迁移建议。

Fusion canonical slug 使用固定顺序 `image -> radar -> gps -> lidar -> mmwave`，覆盖所有 2 到 5 模态组合。例如：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-run-deepsense6g-top8-selector --config configs/deepsense6g_top8_selector.yaml
```

包含 image 或 LiDAR 的 canonical fusion strong 配置使用 `modular_sequence`；默认 lightweight 配置使用 `cls_token_transformer_fusion`。Fusion virtual config 只生成 `strong` 和 `lightweight` 主线；旧 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml` 会失败并提示使用当前入口。

已退役的 HiST-Beam、history-anchored Hist、P3/V7/V8/V9 probe 和默认 LOSO plan 不自动生成 KD variant，也不会由 virtual config alias 接管。

## Snapshot Next-Frame

Snapshot baseline 是 optional/supporting workflow，用于隔离历史窗口收益，不是当前 few-shot cross-scene 主结论的默认步骤。输入只取当前帧 `seq_len=1`，监督只取下一帧 `num_pred=1`，模型 core 为 `snapshot_frame`。

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/snapshot_next_frame_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/all_modalities_snapshot_next_frame_supervised.yaml
```

单模态入口为 `configs/<image|radar|gps|lidar|mmwave>/snapshot_next_frame_supervised.yaml`；fusion 入口为 `configs/fusion/<canonical_slug>_snapshot_next_frame_supervised.yaml`。

## Objective-Aware Fusion

Objective-aware occlusion、position 和 multitask 是 optional/supporting workflow，不是 MMW GPS v2/BGAM 或 DeepSense residual/Top8/BGAM 的前置步骤。预测目标由 `experiment.objective` 选择，合法值为 `beam`、`occlusion`、`position` 和 `multitask`。保留入口使用 `<slug>_<objective>_supervised.yaml` 命名：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_occlusion_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_position_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_multitask_supervised.yaml
```

`strong_only_<objective>_supervised.yaml` 解析为 `[gps, mmwave]`，`weak_only_<objective>_supervised.yaml` 解析为 `[image, radar, lidar]`，可用于普通模态子集调试。

## CSI Hardening

CSI hardening 主矩阵位于 `configs/csi/hardening_matrix/`，debug 矩阵位于 `configs/csi/hardening_matrix/debug/`。普通 CSI supervised baseline 使用 `configs/csi/supervised.yaml`，medium degraded baseline 使用 `configs/csi/medium_degraded_supervised.yaml`。

常用检查：

```bash
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_matrix_configs_load_and_preserve_contracts -q
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_debug_matrix_configs_load_and_isolate_single_changes -q
```

GPS+CSI 验证矩阵位于 `configs/fusion/csi_hardening_matrix/`，包括 GPS-only、GPS+clean CSI、GPS+slow CSI 和 prioritized warmup 配置。对应合同由 `tests/test_student_configs.py::test_gps_csi_validation_matrix_configs_load` 覆盖。

## Raymobtime

Raymobtime s008 是独立数据集家族，默认根目录为 `dataset/Raymobtime/s008`，任务语义为 current snapshot beam selection。预处理、训练、评估和 sensing-only / sensing+ray 实验边界见 [Raymobtime_s008_selection.md](Raymobtime_s008_selection.md)。

## 运行产物

训练输出默认写入 `outputs/<scene_slug>/<run_name>/`，包括 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`metrics.json`、checkpoint、TensorBoard 和可选 normalization/target artifact。使用 virtual/overlay config 时，运行产物仍保存完整解析配置，不依赖原始 YAML 文件继续存在。
