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
- TII-VLRG-style Transformer baseline：`configs/fusion/tii_vlrg_transformer_baseline.yaml`
- RMBP-MM missing-modality baseline：`configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml`
- BEV-Fusion 2604：`configs/fusion/experiments/bev_fusion_2604/`
- AMBER-lite missing-modality：`configs/fusion/amber_lite_missing_modality.yaml`
- AMBER full architecture reproduction：`configs/fusion/amber_full_architecture.yaml`
- RBMA missing-modality ablation：`configs/fusion/experiments/rbma_missing_workflow/`
- Scene31 next-round / night-grid local manifests：`configs/scene31/next_round/experiment_manifest.*`、`configs/scene31/night_grid/experiment_manifest.*`
- MMW GPS v2：`configs/mmw_town_gps_adapter_v2.yaml`
- Physics-informed MMW baseline：`configs/fusion/physics_informed_mmw_debug.yaml`、`configs/fusion/physics_informed_mmw_paper_debug.yaml`、`configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml`
- CSI hardening：`configs/csi/hardening_matrix/` 和 `configs/fusion/csi_hardening_matrix/`
- JEPA shortcut benchmark / visual analysis：`configs/diagnostics/*.yaml`

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

TII-VLRG-style Transformer 默认作为本仓库本地可训练 baseline 运行，不依赖 TII 官方源码、权重或 checkpoint：

```bash
conda run -n kd_mm_beam kd-sensing-train \
  --config configs/fusion/tii_vlrg_transformer_baseline.yaml
```

旧 TII external wrapper 只用于可选导入外部 repo/checkpoint/prediction 的结果，不作为本地 baseline 训练前置。缺 repo/checkpoint/prediction 时只写 `pending` 或 `unavailable`：

```bash
conda run -n kd_mm_beam kd-sensing-tii-vlrg-transformer \
  --config configs/baselines/tii_vlrg_transformer_reproduction.yaml \
  --dry-run \
  --output-root outputs/analysis/tii_vlrg_transformer_reproduction
```

确认外部 repo、checkpoint 和 prediction/metrics 输出路径后才使用 `--execute`；stdout/stderr、manifest、prediction、metrics 和 logs 仍限定在 ignored output root。

RMBP-MM missing-modality baseline 默认作为本地五模态可训练 baseline 运行，使用 image/radar/GPS/LiDAR/mmWave `modular_sequence`、token fusion 和训练期 `modality_dropout` difficulty profile：

```bash
conda run -n kd_mm_beam kd-sensing-train \
  --config configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml
```

WCL source audit 仍可作为可选背景命令，但不再是项目 baseline 的前置：

```bash
conda run -n kd_mm_beam kd-sensing-wcl2025-missing-modality-audit \
  --output-root outputs/analysis/wcl2025_missing_modality_reproduction
```

这些本地 baseline 的 condition-level summary 仍应记录 split、sample_count、label_space、metric_profile、difficulty_digest 和 seed；字段缺失时不要升级为正式结果 claim。

AMBER full architecture reproduction 默认使用同一个训练入口，不新增专用 runner：

```bash
conda run -n kd_mm_beam kd-sensing-train \
  --config configs/fusion/amber_full_architecture.yaml
```

该配置只声明本地 architecture reproduction；缺真实 strict comparable metrics、官方源码/权重或完整评估证据时，claim 保持 `pending` / `unverified`。

RBMA missing-modality workflow 现在默认先跑 weighted_sum / AMBER-style mask 主线，RBMA 仅保留对照；所有配置都走现有训练入口，默认单任务运行：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/amber_style_mask_baseline_fullrun.yaml --auto-resume
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/weighted_sum_mask.yaml --auto-resume
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability.yaml --auto-resume
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability_beam_proto.yaml --auto-resume
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/weighted_sum_reliability_beam_proto_kd.yaml --auto-resume
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/rbma_missing_workflow/no_jepa_rbma_proto_kd_fullrun.yaml --auto-resume
```

批量运行时使用顺序 runner，默认 `--max-parallel 1`：

```bash
conda run -n kd_mm_beam python scripts/run_rbma_missing_workflow.py --auto-resume --num-workers 2
```

Strong-encoder RBMA 和 M2Beam single-modal Scene31 overlay 只保留为 local/manual checkpoint-placeholder 输入；如需人工复跑，直接重复传入 `--config <yaml>` 或运行对应 `kd-sensing-train --config <yaml>`，不再维护固定四 GPU queue shell。

Scene31 night-grid / next-round 是 manifest-backed local/manual 队列，不是长期 package CLI。源码维护 `configs/scene31/night_grid/experiment_manifest.*`、`configs/scene31/next_round/experiment_manifest.*`、base config、generator、P0 fresh eval runner 和 summary helper；实体 YAML 需要本地生成后再用 `kd-sensing-train --config <generated-yaml>` 运行，analysis 输出限定在 ignored `outputs/scene31/analysis/` 和 `outputs/scene31_next_round/`。

P0 训练完成后，完整 fresh eval 与新主指标汇总使用：

```bash
bash scripts/run_scene31_p0_fresh_eval.sh --root outputs/scene31_next_round --gpus 4,5,6,7
conda run -n kd_mm_beam python scripts/summarize_scene31_p0_fresh_eval.py \
  --root outputs/scene31_next_round \
  --out outputs/scene31_next_round/p0_fresh_summary
```

`scripts/run_scene31_p0_fresh_eval.sh --include-baselines` 会额外尝试 `amr_net_supervised` 和 `amber_full_architecture` local baseline；只有对应 checkpoint 已在指定 root 下可解析时才适合作为同批 fresh eval 输入。P0 winner selection 以 `avg_missing -> full -> overall_mean -> balanced` 为默认排序，`balanced` 只作为辅助表。

pattern evaluation 会随训练结束写入 `outputs/scene31/eval/*_missing_patterns.csv/json`；也可手动复用包内 eval matrix：

```bash
conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix \
  --config configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml \
  --checkpoint outputs/.../checkpoints/best.pth \
  --output-dir outputs/eval/rbma_missing_workflow \
  --patterns full missing_gps non_gps_only only_gps random_0.5
```

## Retired AMR-Net_gps_image Tombstone

AMR-Net_gps_image / IEEE `11282996` source-audit mock runner 已退役，不再提供 current CLI、实体配置、mock metrics 或 claim 占位。历史背景只保留 metadata conflict caveat：公开 document `11282996` 与 DeepSense6G Scenario 23 作者包 document `10000718` 不一致；旧本地产物不能声明 official reproduction。当前 GPS+Image 对照使用 Vision-Position suite 或 Arnold22 Camera AE+GPS Direct。

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

Predictive hybrid 是独立的 BeamBench-fair 派生线，不等同 GPS-query pooling baseline。训练入口启用 `hybrid_residual_query`、temporal auxiliary branch、`feature_consistency_gate` 和 `seq_len=4` / `history_window=3`，训练 difficulty profile 默认使用 legacy `P4_joint_predictive_recovery`。单个 `P4_joint_predictive_recovery` train/curriculum profile 不等价于完整 clean + `image_missing` / `image_noise` / `gps_noise` stress-curve benchmark，也不产生真实数值 claim：

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_predictive_hybrid_beambench_fair_lowmem.yaml
```

Predictive robustness smoke 只验证 stress-curve schema、strict comparability、margin-vs-ResNet 和输出 manifest，不产生真实性能 claim：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_predictive_robustness_smoke.yaml \
  --output-dir outputs/analysis/predictive_jepa_robustness/smoke \
  --force
```

真实 train-then-evaluate 需要先训练并登记本地 checkpoint provenance，再用本地派生 manifest 替换 smoke manifest 中的 `synthetic_metrics`、mock weights 和 `allow_missing_artifacts`，并提供 clean anchor、默认 stress curves condition-level metrics、strict comparability fields、difficulty digest、seed、split、sample_count 和 Image ResNet+GPS baseline。真实运行产物仍写入 ignored `outputs/analysis/predictive_jepa_robustness/`：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest outputs/analysis/predictive_jepa_robustness/real/manifest.yaml \
  --output-dir outputs/analysis/predictive_jepa_robustness/real \
  --force
```

## Retired JEPA-MSAC Scenario 32 Tombstone

JEPA-MSAC Scenario 32 mock/paper workflow 已退役，不再提供 current CLI、pretraining config、whole-model registry surface、loss、objective 或 focused smoke。历史说明只作为 tombstone 保留；当前 JEPA 相关工作使用 GPS-conditioned JEPA、JEPA visual analysis、GPS shortcut benchmark 或仍维护的 JEPA downstream configs。

## BEV-Fusion 2604

```bash
conda run -n kd_mm_beam pytest tests/test_bev_fusion_2604.py -q
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/low_memory.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/paper_full.yaml
```

`paper_full.yaml` is the formal 2604-aligned protocol. `low_memory.yaml` is a paper approximation and `smoke.yaml` is synthetic/mock schema validation only. Ablations under `configs/fusion/experiments/bev_fusion_2604/ablations/` inherit the same split and must be reported by `ablation_name`.

## AMBER-lite Missing-Modality

AMBER-lite 是本地实验 baseline，不是完整 AMBER 官方复现。训练入口复用 `modular_sequence`，默认不下载外部权重，对 image/radar/GPS/LiDAR 做训练期 modality dropout，真实运行产物只写入 ignored `outputs/analysis/local_baselines/amber_lite_missing_modality/`：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/amber_lite_missing_modality.yaml
```

评估 suite manifest 覆盖 clean、单模态缺失、多模态缺失、poor image、LiDAR/radar unavailable、wrong/async GPS。缺真实 metrics、LiDAR/radar artifact 或 strict comparability 字段时，summary row 必须保持 `pending`、`unavailable` 或 `not_comparable`，不得进入 strict ranking：

```bash
configs/diagnostics/amber_lite_missing_modality_eval.yaml
```

## MMW, CSI, Diagnostics

MMW GPS v2:

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --config configs/mmw_town_gps_adapter_v2.yaml --label-space mapping_enabled --save-logits --save-prior-probs
conda run -n kd_mm_beam kd-sensing-plot-mmw-town-gps-v2 --results-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled
conda run -n kd_mm_beam kd-sensing-compare-mmw-town-gps-v2 --previous-dir outputs/analysis/mmw_town_label_distribution --new-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled
```

Physics-informed MMW baseline uses the same training entry and a package inspection command:

```bash
conda run -n kd_mm_beam kd-sensing-inspect-mmw-physics --config configs/fusion/physics_informed_mmw_debug.yaml --max-samples 1
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_vision_only.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_partial_csi_multimodal.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_history_csi_multimodal.yaml
```

Use `physics_informed_mmw_no_physics.yaml`、`physics_informed_mmw_no_csi_reconstruction.yaml`、`physics_informed_mmw_no_path_loss.yaml`、`physics_informed_mmw_no_array_consistency.yaml`、`physics_informed_mmw_no_physics_head.yaml`、`physics_informed_mmw_csi_only.yaml`、`physics_informed_mmw_image_only.yaml`、`physics_informed_mmw_image_csi.yaml`、`physics_informed_mmw_full_multimodal.yaml` 和 `physics_informed_mmw_oracle_full_csi.yaml` for ablations/upper-bound checks. Current local result summary: sparse CSI provides the main multimodal gain, task-aligned PINN gives modest Top-1 improvement, array consistency is the strongest useful physics term, and raw CSI reconstruction is negative transfer. Current full CSI is `csi_target` and is not a default model input; only `sparse_pilot`/`history`/`partial`/`noisy`/`compressed` are leakage-safe `csi_input` modes. The `oracle_full` config is explicitly upper-bound only and stays outside MMW sensor-assisted main claims.

CSI hardening:

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/csi/hardening_matrix/debug/A0_original.yaml
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_csi_hardening_matrix_configs_load_and_preserve_contracts -q
conda run -n kd_mm_beam pytest tests/test_student_configs.py::test_gps_csi_validation_matrix_configs_load -q
```

JEPA shortcut benchmark and visual analysis:

```bash
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml \
  --output-dir outputs/analysis/jepa_gps_shortcut_benchmark/smoke \
  --force

conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml \
  --output-dir outputs/analysis/scenario_d_image_observability/smoke \
  --force

conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_predictive_robustness_smoke.yaml \
  --output-dir outputs/analysis/predictive_jepa_robustness/smoke \
  --force

conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis \
  --analysis-config configs/diagnostics/jepa_visual_analysis_2604.yaml \
  --output-dir outputs/visual_analysis/jepa_query_pool_2604 \
  --force
```

Scenario D smoke writes only ignored local artifacts: legacy `results/scenario_d_image_observability.csv` / `results/heatmap_cx_dy.npy` plus CxD phase, dominance-status, crossing and failure-decomposition outputs. Synthetic dominance rows remain mock/unavailable unless the manifest points to real gradient, attention/fusion weight or latent diagnostics.

Difficulty profiles under `configs/difficulty/` are training/evaluation reliability profiles, not new modalities. They may perturb input tensors and reliability metadata, but must not move `target_beam`, soft targets or split metadata.

## 已退役边界

HiST-Beam、history-anchored Hist、Raymobtime s008、standalone Top8 selector、GPS coarse anchor、GPS residual、camera residual、BGAM、viewer manifest、Gradio viewer、AMR-Net_gps_image、JEPA-MSAC、固定 shell orchestration、MMW GPS v2 旁支 `scripts/mmw/visualize_gps_*`、CRAF/MARF/G2D、Multimodal-NF 和旧 KD/Fusion KD 路线不再作为当前入口维护。旧配置、CLI、registry 名称或 historical output 只能作为退役、历史或 migration guard 说明出现。
