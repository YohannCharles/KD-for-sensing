# 实验矩阵

本文件承载 README 中移出的实验矩阵和推荐运行顺序。命令默认使用 `kd_mm_beam` 环境；训练、评估和预处理优先使用 console script。

## 单模态和基础 Fusion

单模态 canonical 矩阵保留强模型和轻量模型 no-KD baseline。KD 配置仍可运行，但只作为 legacy/optional supplemental baseline：

| 模态 | 强模型 no-KD baseline | 轻量 no-KD baseline | legacy KD supplemental |
| --- | --- | --- | --- |
| image | `configs/image/teacher_no_kd.yaml` | `configs/image/student_no_kd.yaml` | `configs/image/logits_kd.yaml`, `configs/image/rkd.yaml` |
| radar | `configs/radar/teacher_no_kd.yaml` | `configs/radar/student_no_kd.yaml` | `configs/radar/logits_kd.yaml`, `configs/radar/rkd.yaml` |
| gps | `configs/gps/teacher_no_kd.yaml` | `configs/gps/student_no_kd.yaml` | `configs/gps/logits_kd.yaml`, `configs/gps/rkd.yaml` |
| lidar | `configs/lidar/teacher_no_kd.yaml` | `configs/lidar/student_no_kd.yaml` | `configs/lidar/logits_kd.yaml`, `configs/lidar/rkd.yaml` |
| mmwave | `configs/mmwave/teacher_no_kd.yaml` | `configs/mmwave/student_no_kd.yaml` | `configs/mmwave/logits_kd.yaml`, `configs/mmwave/rkd.yaml` |

推荐主线顺序是先运行 no-KD supervised / adaptation baseline，再进入 HiST-Beam、MMW sensor-assisted、history-anchored residual、adapter/prototype/calibration、Raymobtime s008、CSI hardening 或 viewer manifest。`logits_kd` 和 `rkd` 不再是 quickstart 或主结论必需步骤；显式运行单模态实体配置时会写出 `method_family=legacy_kd`、`distillation_enabled=true`、`baseline_role=optional_baseline` 和 `reproduction_scope=historical_reproduction`，summary 默认把它们作为 supplemental comparison。

Fusion canonical slug 使用固定顺序 `image -> radar -> gps -> lidar -> mmwave`，覆盖所有 2 到 5 模态组合。例如：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_student_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_student_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-hist-beam-loso --config configs/hist_beam/quick_smoke.yaml
```

包含 image 或 LiDAR 的 canonical fusion teacher/no-KD 配置使用 `modular_sequence`；默认 student 使用 `cls_token_transformer_fusion`。Fusion virtual config 只生成 `teacher_no_kd` 和 `student_no_kd` 主线；不存在实体 YAML 的 `<slug>_logits_kd.yaml` 或 `<slug>_rkd.yaml` 会失败并提示 legacy KD fusion virtual alias 已退役。需要历史 KD 对照时，使用仍被跟踪的单模态实体配置，或通过独立 baseline change 增加显式 fusion 实体配置。

HiST-Beam、MMW sensor-assisted quick validation、history-anchored residual quick validation 和默认 LOSO plan 不自动生成 KD variant。KD baseline 若后续用于 HiST-Beam 对照，应以单独 profile 或显式配置进入，并保持 supplemental/legacy 分组。

## Snapshot Next-Frame

Snapshot baseline 是 optional/supporting workflow，用于隔离历史窗口收益，不是当前 few-shot cross-scene 主结论的默认步骤。输入只取当前帧 `seq_len=1`，监督只取下一帧 `num_pred=1`，模型 core 为 `snapshot_frame`。

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/snapshot_next_frame_no_kd.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/all_modalities_snapshot_next_frame_no_kd.yaml
```

单模态入口为 `configs/<image|radar|gps|lidar|mmwave>/snapshot_next_frame_no_kd.yaml`；fusion 入口为 `configs/fusion/<canonical_slug>_snapshot_next_frame_no_kd.yaml`。

## Objective-Aware Fusion

Objective-aware occlusion、position 和 multitask 是 optional/supporting workflow，不是 HiST-Beam/MMW target adaptation 的前置步骤。预测目标由 `experiment.objective` 选择，合法值为 `beam`、`occlusion`、`position` 和 `multitask`。保留入口使用 `<slug>_<objective>_no_kd.yaml` 命名：

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
