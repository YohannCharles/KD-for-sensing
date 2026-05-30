# 实验矩阵

本文件承载 README 中移出的实验矩阵和推荐运行顺序。命令默认使用 `kd_mm_beam` 环境；训练、评估和预处理优先使用 console script。

## 单模态和基础 Fusion

单模态 canonical 矩阵：

| 模态 | Teacher baseline | Student baseline | KD |
| --- | --- | --- | --- |
| image | `configs/image/teacher_no_kd.yaml` | `configs/image/student_no_kd.yaml` | `configs/image/logits_kd.yaml`, `configs/image/rkd.yaml` |
| radar | `configs/radar/teacher_no_kd.yaml` | `configs/radar/student_no_kd.yaml` | `configs/radar/logits_kd.yaml`, `configs/radar/rkd.yaml` |
| gps | `configs/gps/teacher_no_kd.yaml` | `configs/gps/student_no_kd.yaml` | `configs/gps/logits_kd.yaml`, `configs/gps/rkd.yaml` |
| lidar | `configs/lidar/teacher_no_kd.yaml` | `configs/lidar/student_no_kd.yaml` | `configs/lidar/logits_kd.yaml`, `configs/lidar/rkd.yaml` |
| mmwave | `configs/mmwave/teacher_no_kd.yaml` | `configs/mmwave/student_no_kd.yaml` | `configs/mmwave/logits_kd.yaml`, `configs/mmwave/rkd.yaml` |

推荐顺序是 `teacher_no_kd -> student_no_kd -> logits_kd/rkd`。KD 配置优先从当前 scene 的 checkpoint registry 读取同模态 teacher no-KD 最佳权重；也可以通过 `distillation.teacher_model_name` 或评估入口 `--weights` 显式指定。

Fusion canonical slug 使用固定顺序 `image -> radar -> gps -> lidar -> mmwave`，覆盖所有 2 到 5 模态组合。例如：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_teacher_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_student_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_logits_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_rkd.yaml
```

包含 image 或 LiDAR 的 canonical fusion teacher/no-KD 配置使用 `modular_sequence`；默认 student/KD student 使用 `cls_token_transformer_fusion`。需要复现实验中的 early-concat 或旧 token transformer baseline 时，使用对应显式配置路径；退役研究线的配置路径不会被 virtual alias 接管。

## Snapshot Next-Frame

Snapshot baseline 用于隔离历史窗口收益：输入只取当前帧 `seq_len=1`，监督只取下一帧 `num_pred=1`，模型 core 为 `snapshot_frame`。

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/snapshot_next_frame_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml
```

单模态入口为 `configs/<image|radar|gps|lidar|mmwave>/snapshot_next_frame_no_kd.yaml`；fusion 入口为 `configs/fusion/<canonical_slug>_snapshot_next_frame_no_kd.yaml`。

## Objective-Aware Fusion

预测目标由 `experiment.objective` 选择，合法值为 `beam`、`occlusion`、`position` 和 `multitask`。推荐 objective-aware fusion 入口使用 `<slug>_<objective>_no_kd.yaml` 命名：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_occlusion_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_position_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml
```

`strong_only_<objective>_no_kd.yaml` 解析为 `[gps, mmwave]`，`weak_only_<objective>_no_kd.yaml` 解析为 `[image, radar, lidar]`，可用于普通模态子集调试。

## CSI Hardening

CSI hardening 主矩阵位于 `configs/csi/hardening_matrix/`，debug 矩阵位于 `configs/csi/hardening_matrix/debug/`。本变更只做 inventory，不删除这些实体配置。

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
