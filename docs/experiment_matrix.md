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

推荐主线顺序是先运行 supervised/adaptation baseline，再进入 DeepSense6G/MMW GPS+LiDAR BGAM、MMW GPS v2、CSI hardening 或 viewer manifest。旧 `teacher_no_kd`、`student_no_kd`、`no_kd`、`logits_kd`、`rkd`、`configs/hist_beam/*`、HiST-Beam、GPS coarse anchor、Top8 selector、GPS residual、camera residual 和 Raymobtime s008 入口不再作为支持入口存在；配置加载器会拒绝这些路径并给出迁移或退役说明。

Fusion canonical slug 使用固定顺序 `image -> radar -> gps -> lidar -> mmwave`，覆盖所有 2 到 5 模态组合。例如：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-run-deepsense6g-gps-lidar-bgam --config configs/deepsense6g_gps_lidar_bgam.yaml
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

Objective-aware occlusion、position 和 multitask 是 optional/supporting workflow，不是 MMW GPS v2/BGAM 或 DeepSense BGAM 的前置步骤。预测目标由 `experiment.objective` 选择，合法值为 `beam`、`occlusion`、`position` 和 `multitask`。保留入口使用 `<slug>_<objective>_supervised.yaml` 命名：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_occlusion_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_position_supervised.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_multitask_supervised.yaml
```

`strong_only_<objective>_supervised.yaml` 解析为 `[gps, mmwave]`，`weak_only_<objective>_supervised.yaml` 解析为 `[image, radar, lidar]`，可用于普通模态子集调试。

## GPS-conditioned JEPA 预训练

GPS-conditioned JEPA 是 image+GPS 自监督预训练入口，使用 `experiment.objective: gps_conditioned_jepa` 和 `model.primary.type: gps_conditioned_jepa`。canonical smoke 配置位于：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/pretraining/deepsense6g_gps_conditioned_jepa_smoke.yaml
```

完整主实验使用 paper-split 风格的 low-memory 配置：训练拼接 DeepSense6G scenes 32、33、34，验证/监控覆盖 scenes 31、32、33、34。该配置使用 1 个训练 worker、0 个验证 worker、关闭 persistent worker/pinned memory，并默认读取已预热的 RGB/ImageNet derived cache，避免 full split 训练时 DataLoader worker 常驻内存被放大：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/deepsense6g_s31_image_derived_cache.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/deepsense6g_s32_image_derived_cache.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/deepsense6g_s33_image_derived_cache.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/deepsense6g_s34_image_derived_cache.yaml

conda run -n kd_mm_beam kd-sensing-train --config configs/pretraining/deepsense6g_gps_conditioned_jepa_full_lowmem.yaml

conda run -n kd_mm_beam kd-sensing-train --config configs/pretraining/deepsense6g_gps_conditioned_jepa_gps_biased_lowmem.yaml
```

如果显存仍明显空闲，可优先只调大 batch size；如果显存 OOM，则把两个配置的 `data.dataloader.train_batch_size` 和 `test_batch_size` 从 64 降到 32。不要重新启用 `persistent_workers=true` 或把 worker 数量一次性加回 4；这会重新放大 CPU RAM 占用。

该入口使用 RGB/ImageNet image profile 与 GPS relative-polar 特征，只记录 `val_jepa_loss`、JEPA mask ratio、EMA decay 和通用 loss；不会计算 beam Top-K、DBA、occlusion、position、LOS 或 link 指标。多场景主实验运行产物写入 `outputs/<run_name>/`，checkpoint 保存完整 `model.primary`，`runtime.prediction_objective.jepa.context_encoder_artifact_key` 标明可复用的 context encoder state-dict key。该 checkpoint 可作为后续 fine-tuning change 的初始化来源，但本入口不自动改写 supervised beam/fusion 配置，也不恢复旧 KD/teacher 体系。

### JEPA 下游复用公平复核

和 BeamBench Table III 做下游指标复核时，使用 fair low-memory 配置族，而不是 scene31-only 或 `num_pred=3` 的快速调试配置。fair 配置训练 scenes 32、33、34，从训练 split 内部划分 validation 做 early stopping/checkpoint selection，训练结束后单独加载 `best.pth` 在 scenes 31、32、33、34 的 test split 上记录 `final_test_metrics`。该配置固定 prediction window 为 `num_pred=1`，保留当前 image+GPS supervised 的 `seq_len=8`，DBA 距离口径设为 BeamBench linear，scheduler 设为 `none`。这些配置属于 JEPA image+GPS 实验复现面，路径位于 `configs/fusion/experiments/jepa_image_gps/`，不作为 `configs/fusion/` 根目录推荐入口。

当前 Image+GPS+JEPA 下游主线是 GPS-biased checkpoint reuse，即 `image_gps_jepa_gps_biased_best_*` 配置族。supervised 与 random-mask 配置只作为对照；next-beam query/plain-token/GRU/snapshot 系列只作为 ablation，除非在同一评价协议上超过 GPS-biased 主线，否则不替代主结论。

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml
```

JEPA random 配置默认复用 `outputs/deepsense6g_gps_conditioned_jepa_full_s32_s34_lowmem/checkpoints/{best,last}.pth`，GPS-biased 配置默认复用 `outputs/deepsense6g_gps_conditioned_jepa_gps_biased_s32_s34_lowmem/checkpoints/best.pth`。

### 2604.05668 S32-34 对齐复核

和 arXiv:2604.05668 的主表比较时，使用 2604 对齐配置族，而不是 BeamBench-fair 配置。该配置族合并 DeepSense6G scenes 32、33、34 的官方 train/test labeled CSV，并在每个 scene 内按 `future_beam1` 标签固定 seed 做 `80/10/10` stratified train/validation/test split；历史窗口改为 `seq_len=5`，预测窗口保持 `num_pred=1`，DBA 距离口径为 linear。该口径不评估 scene31 泛化，最终报告 S32/S33/S34 test DBA 和三场景宏平均。

2604-style 主报告使用 `image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml` 及其 `best.pth`。当前本地复核的主线结果为：S32/S33/S34 DBA `0.8777 / 0.8853 / 0.8796`，macro DBA `0.8809`。写作时表述为“在我们复现的 2604-style stratified 80/10/10 split 上，Image+GPS + JEPA gps-biased 达到 88.09% macro DBA，高于论文 BEV-Fusion 报告的 86.52%”；不要写成严格证明超过原论文 exact split，因为原文未释放 exact split index/seed。

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml
```

当前真实数据构建该 split 时，S32/S33/S34 合计 11015 条样本，切分为 train 8839、validation 1088、test 1088；每个 split 的 scene 组成和 scaler 来源会写入 `final_config.yaml` runtime metadata。

## CSI Hardening

CSI hardening 主矩阵位于 `configs/csi/hardening_matrix/`，debug 矩阵位于 `configs/csi/hardening_matrix/debug/`。普通 CSI supervised baseline 使用 `configs/csi/supervised.yaml`，medium degraded baseline 使用 `configs/csi/medium_degraded_supervised.yaml`。

常用检查：

```bash
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_matrix_configs_load_and_preserve_contracts -q
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_debug_matrix_configs_load_and_isolate_single_changes -q
```

GPS+CSI 验证矩阵位于 `configs/fusion/csi_hardening_matrix/`，包括 GPS-only、GPS+clean CSI、GPS+slow CSI 和 prioritized warmup 配置。对应合同由 `tests/test_student_configs.py::test_gps_csi_validation_matrix_configs_load` 覆盖。

## 已退役：Raymobtime s008

Raymobtime s008 的 dataset type、预处理配置、selection 模型、`coord/ray` 模态和 focused tests 已退役并从当前矩阵删除。旧 `configs/raymobtime/*`、`configs/preprocess/raymobtime_s008_*.yaml`、`raymobtime_s008`、`simple_concat_multitask_selection`、`task_aware_gated_multitask_selection` 和 `raymobtime_lidar_3d_cnn` 只保留快速失败提示，不再作为推荐入口。

## 运行产物

训练输出默认写入 `outputs/<scene_slug>/<run_name>/`，包括 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`metrics.json`、checkpoint、TensorBoard 和可选 normalization/target artifact。使用 virtual/overlay config 时，运行产物仍保存完整解析配置，不依赖原始 YAML 文件继续存在。
