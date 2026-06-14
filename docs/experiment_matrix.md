# 实验矩阵 Quickstart

本文件只保留推荐顺序、入口命令和关键 caveat。完整横向表格已经转移到：

- 当前主线模型目录：[docs/mainline_model_catalog.md](mainline_model_catalog.md)
- 实验协议和参数口径：[docs/experiment_protocols.md](experiment_protocols.md)
- 结果和 claim 账本：[docs/result_claims_registry.md](result_claims_registry.md)

命令默认使用 `kd_mm_beam` 环境；训练、评估和预处理优先使用 console script。所有真实训练、metrics、figures、checkpoint、feature cache 和日志都写入 ignored 的 `outputs/`、`outputs/cache/` 或 `logs/`，不进入源码变更。

## 推荐顺序

1. 先跑最小健康检查和配置加载：

```bash
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q
```

2. 建立 supervised/adaptation 或 paired control 基线：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_gps_supervised.yaml
```

3. 选择一个 current 主线 family，并只在同 family 内比较：

- Image+GPS JEPA BeamBench-fair：`configs/fusion/experiments/jepa_image_gps/*beambench_fair_lowmem.yaml`
- Image+GPS JEPA 2604-style：`configs/fusion/experiments/jepa_image_gps/*2604_s32_s34_lowmem.yaml`
- Arnold22 Camera AE+GPS Direct：`configs/fusion/beambench_image_ae_gps_direct.yaml` + 专用 Table III runner
- BEV-Fusion 2604：`configs/fusion/experiments/bev_fusion_2604/`
- DeepSense6G/MMW BGAM：`configs/deepsense6g_gps_lidar_bgam.yaml`、`configs/mmw_town_gps_lidar_bgam.yaml`
- MMW GPS v2：`configs/mmw_town_gps_adapter_v2.yaml`
- CSI hardening：`configs/csi/hardening_matrix/` 和 `configs/fusion/csi_hardening_matrix/`
- JEPA shortcut benchmark / visual analysis / viewer manifest：`configs/diagnostics/*.yaml`

## 单模态和基础 Fusion

单模态 canonical 矩阵使用 strong、lightweight 和 supervised 三类入口。所有入口都构建单个 `model.primary` 主模型。

| 模态 | strong | lightweight | supervised |
| --- | --- | --- | --- |
| image | `configs/image/strong.yaml` | `configs/image/lightweight.yaml` | `configs/image/supervised.yaml` |
| radar | `configs/radar/strong.yaml` | `configs/radar/lightweight.yaml` | `configs/radar/supervised.yaml` |
| gps | `configs/gps/strong.yaml` | `configs/gps/lightweight.yaml` | `configs/gps/supervised.yaml` |
| lidar | `configs/lidar/strong.yaml` | `configs/lidar/lightweight.yaml` | `configs/lidar/supervised.yaml` |
| mmwave | `configs/mmwave/strong.yaml` | `configs/mmwave/lightweight.yaml` | `configs/mmwave/supervised.yaml` |

Fusion canonical slug 使用固定顺序 `image -> radar -> gps -> lidar -> mmwave`，覆盖合法 2 到 5 模态组合。例如：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_lightweight.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml
```

旧 `teacher_no_kd`、`student_no_kd`、`no_kd`、`logits_kd`、`rkd`、`configs/hist_beam/*`、HiST-Beam、GPS coarse anchor、Top8 selector、GPS residual、camera residual 和 Raymobtime s008 入口已经退役或只作为 migration guard；配置加载器会拒绝这些路径并给出迁移或退役说明。

## Vision-Position 和 Arnold22

严格官方 BeamBench GPS `Classical*` / `Dense†` 仍需要官方 BeamBench repo、官方 test CSV、官方权重和官方环境。本仓库的 Vision-Position presets 只是项目 neural/control baseline：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/resnet_gps.yaml \
  -o data.dataset.train_scenes=[32,33,34] \
  -o data.dataset.test_scenes=[31,32,33,34] \
  -o data.validation_from_train.enabled=true
```

Arnold22 Table III `Camera=AE, GPS=Direct, Fusion=Yes` 当前本地 substitute 只使用 current beam selection、`seq_len=1`、`num_pred=1`、GPS `paper_distance_angle`、scene paper calibration angle 和 linear/non-circular DBA：

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --train-scenes 32 33 34 \
  --eval-scenes 31 32 33 34 \
  --selection-split validation \
  --gps-feature-mode paper_distance_angle \
  --target-beam-source current \
  --output-root outputs/scenegroup_s32_s34/beambench_image_ae_gps_direct_tableiii/beambench_aligned
```

缺 official AE/fusion 权重、official exact test packaging 或官方完整训练搜索流程时，claim status 必须是 `local substitute`、`local strict-validation`、`blocked official reproduction` 或 `upper-bound`，不得写成 official reproduction。旧 `--target-beam-source future` 记录只作为 historical sequence-prediction ablation，不是当前 Table III strict setup。

## Image+GPS JEPA

BeamBench-fair family 用于下游输入/split/target/metric 对齐，不是 Table III Camera AE+GPS Direct 模型：

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml
```

2604-style family 用于 S32/S33/S34 stratified 80/10/10 split 对齐：

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml
```

GPS-query pooling configs must be paired against the matching GPS-biased mean-pooling baseline from the same family. Do not mix BeamBench-fair and 2604-style checkpoints, label spaces, split protocols or schedules.

## BEV-Fusion 2604

```bash
conda run -n kd_mm_beam pytest tests/test_bev_fusion_2604.py -q
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/low_memory.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/paper_full.yaml
```

`paper_full.yaml` is the formal 2604-aligned protocol. `low_memory.yaml` is a paper approximation and `smoke.yaml` is synthetic/mock schema validation only. Ablations under `configs/fusion/experiments/bev_fusion_2604/ablations/` inherit the same split and must be reported by `ablation_name`.

## BGAM, MMW, CSI, Diagnostics

DeepSense6G BGAM:

```bash
conda run -n kd_mm_beam kd-sensing-prepare-deepsense6g-gps-lidar-bgam-manifest --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8
conda run -n kd_mm_beam kd-sensing-run-deepsense6g-gps-lidar-bgam --config configs/deepsense6g_gps_lidar_bgam.yaml --support-ratio 0.15 --label-space mapping_disabled --topk 8
```

MMW GPS v2 and MMW BGAM:

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --config configs/mmw_town_gps_adapter_v2.yaml --label-space mapping_enabled --save-logits --save-prior-probs
conda run -n kd_mm_beam kd-sensing-run-mmw-town-gps-lidar-bgam --config configs/mmw_town_gps_lidar_bgam.yaml --label-space mapping_enabled --topk 8
```

CSI hardening:

```bash
NEW_RUN=1 conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh debug
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_matrix_configs_load_and_preserve_contracts -q
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_gps_csi_validation_matrix_configs_load -q
```

JEPA shortcut benchmark and visual analysis:

```bash
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml \
  --output-dir outputs/analysis/jepa_gps_shortcut_benchmark/smoke \
  --force

conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis \
  --analysis-config configs/diagnostics/jepa_visual_analysis_2604.yaml \
  --output-dir outputs/visual_analysis/jepa_query_pool_2604 \
  --force
```

Viewer manifest:

```bash
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/cache/diagnostics/viewer_manifest \
  --scenes 32
```

Difficulty profiles under `configs/difficulty/` are training/evaluation reliability profiles, not new modalities. They may perturb input tensors and reliability metadata, but must not move `target_beam`, soft targets or split metadata.

## 已退役边界

HiST-Beam、history-anchored Hist、Raymobtime s008、standalone Top8 selector、GPS coarse anchor、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 和旧 KD/Fusion KD 路线不再作为当前入口维护。旧配置、CLI、registry 名称或 historical output 只能作为退役、历史或 migration guard 说明出现。
