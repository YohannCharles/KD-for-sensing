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

## Vision-Position Baselines

Arnold22 BeamBench Table III 的严格可比实验只包含论文定义的模型行。GPS-only 不能用本仓库的 `gps_only_neural` 顶替：论文 `Classical*` 是 calibrated GPS angle 的 least-square 规则，`Dense†` 是官方 `dense_model` + `config/gps_dense.cfg` 流程；二者都不是 LSTM/MLP 分类器。

严格复现 GPS `Classical*` 和 `Dense†` 时，使用官方 BeamBench repo、官方 test CSV、官方 `results/models` 权重/校正文件：

```bash
conda run -n kd_mm_beam python scripts/eval_baseline.py \
  --classical-gps \
  --official-root /path/to/BeamBench \
  --data-root /path/to/BeamBench/raw_data/test \
  --csv ml_challenge_test_multi_modal.csv \
  --output-dir outputs/evaluations/beambench_official_gps_classical \
  --execute

conda run -n kd_mm_beam python scripts/eval_baseline.py \
  --official-root /path/to/BeamBench \
  --data-root /path/to/BeamBench/raw_data/test \
  --csv ml_challenge_test_multi_modal.csv \
  --type-list gps_dense \
  --adapt adapt_ \
  --seed 42 \
  --output-dir outputs/evaluations/beambench_official_gps_dense \
  --execute
```

严格复现 Camera=AE, GPS=Direct, Fusion=Yes 行时，优先使用官方权重；本仓库专用 runner 只是在本地 sequence CSV 上的 paper-style substitute，默认 `beam_target_source=current`、`seq_len=1`、`num_pred=1`、GPS `paper_distance_angle`、linear Top-3 DBA：

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

以下 virtual presets 只作为项目对照实验；它们可采用 BeamBench-style 输入、target 和 metric，但不得报告为 Table III 行。配置默认使用 `beam_target_source=current`，DBA 使用 linear/non-circular 距离，Top-K 记录 `1/3/5`。

| preset | config | primary model | Table III 等价性 |
| --- | --- | --- | --- |
| `camera_ae_gps` | `configs/fusion/camera_ae_gps.yaml` | `vision_position_late_fusion` + `camera_ae_frozen` | 否；请用专用 runner |
| `resnet_gps` | `configs/fusion/resnet_gps.yaml` | `vision_position_late_fusion` + `resnet18_imagenet_rgb` | 否；论文没有该行 |
| `transformer_image_gps` | `configs/fusion/transformer_image_gps.yaml` | `vision_position_transformer_fusion` | 否；论文没有该行 |
| `gps_only_neural` | `configs/fusion/gps_only_neural.yaml` | `gps_sequence_baseline` | 否；不是 `Classical*`/`Dense†` |

项目对照实验命令如下，表格标题必须写明 “project neural/control baseline”，不要写成 Arnold22 GPS-only 复现：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/resnet_gps.yaml \
  -o data.dataset.train_scenes=[32,33,34] \
  -o data.dataset.test_scenes=[31,32,33,34] \
  -o data.validation_from_train.enabled=true

conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/transformer_image_gps.yaml \
  -o data.dataset.train_scenes=[32,33,34] \
  -o data.dataset.test_scenes=[31,32,33,34] \
  -o data.validation_from_train.enabled=true

conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/gps_only_neural.yaml \
  -o data.dataset.train_scenes=[32,33,34] \
  -o data.dataset.test_scenes=[31,32,33,34] \
  -o data.validation_from_train.enabled=true
```

单场景产物写入 ignored 的 `outputs/scene<id>/<run_name>/`，多场景产物写入 `outputs/scenegroup_<range-or-list>/<run_name>/`；不要提交真实数据、cache、checkpoint、metrics 或训练日志。

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

该入口使用 RGB/ImageNet image profile 与 GPS relative-polar 特征，只记录 `val_jepa_loss`、JEPA mask ratio、EMA decay 和通用 loss；不会计算 beam Top-K、DBA、occlusion、position、LOS 或 link 指标。多场景主实验运行产物写入 `outputs/scenegroup_s32_s34/<run_name>/`，checkpoint 保存完整 `model.primary`，`runtime.prediction_objective.jepa.context_encoder_artifact_key` 标明可复用的 context encoder state-dict key。该 checkpoint 可作为后续 fine-tuning change 的初始化来源，但本入口不自动改写 supervised beam/fusion 配置，也不恢复旧 KD/teacher 体系。

### JEPA 下游复用公平复核

和 BeamBench Table III 做下游指标复核时，使用 fair low-memory 配置族，而不是 scene31-only 或 `num_pred=3` 的快速调试配置。fair 配置训练 scenes 32、33、34，从训练 split 内部划分 validation 做 early stopping/checkpoint selection，训练结束后单独加载 `best.pth` 在 scenes 31、32、33、34 的 test split 上记录 `final_test_metrics`。该配置固定 `seq_len=1`、`num_pred=1`、`beam_target_source=current`、GPS `paper_distance_angle` 二维输入、scene paper calibration angle、BeamBench linear DBA 和 `1/3/5` Top-K，scheduler 设为 `none`。这些配置属于 JEPA image+GPS 实验复现面，路径位于 `configs/fusion/experiments/jepa_image_gps/`，不作为 `configs/fusion/` 根目录推荐入口。

这里的 fair/BeamBench 对齐是输入、split、target 和 metric 对齐，不是 Table III Camera AE+GPS Direct 模型本身；Table III row 仍以 `configs/fusion/beambench_image_ae_gps_direct.yaml` 专用 runner 为准。

当前 Image+GPS+JEPA 下游主线是 GPS-biased checkpoint reuse 和 GPS-query pooling，即 `image_gps_jepa_gps_biased_best_*` 与 `image_gps_jepa_gps_query_pool_best_*` 配置族。supervised 与 random-mask 配置只作为对照；next-beam query/plain-token/GRU/snapshot 系列已退役删除，必要时只从 git 历史恢复作历史复核，不再作为当前配置矩阵维护。

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_beambench_fair_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml
```

JEPA random 配置默认复用 `outputs/scenegroup_s32_s34/deepsense6g_gps_conditioned_jepa_full_s32_s34_lowmem/checkpoints/{best,last}.pth`，GPS-biased 配置默认复用 `outputs/scenegroup_s32_s34/deepsense6g_gps_conditioned_jepa_gps_biased_s32_s34_lowmem/checkpoints/best.pth`。

### 2604.05668 S32-34 对齐复核

和 arXiv:2604.05668 的主表比较时，使用 2604 对齐配置族，而不是 BeamBench-fair 配置。该配置族合并 DeepSense6G scenes 32、33、34 的官方 train/test labeled CSV，并在每个 scene 内按 `future_beam1` 标签固定 seed 做 `80/10/10` stratified train/validation/test split；历史窗口改为 `seq_len=5`，预测窗口保持 `num_pred=1`，DBA 距离口径为 linear。该口径不评估 scene31 泛化，最终报告 S32/S33/S34 test DBA 和三场景宏平均。

2604-style 主报告使用 `image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml` 及其 `best.pth`。当前本地复核的主线结果为：S32/S33/S34 DBA `0.8777 / 0.8853 / 0.8796`，macro DBA `0.8809`。写作时表述为“在我们复现的 2604-style stratified 80/10/10 split 上，Image+GPS + JEPA gps-biased 达到 88.09% macro DBA，高于论文 BEV-Fusion 报告的 86.52%”；不要写成严格证明超过原论文 exact split，因为原文未释放 exact split index/seed。

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 0-7 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_supervised_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 8-15 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_random_best_2604_s32_s34_lowmem.yaml
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 16-23 conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_2604_s32_s34_lowmem.yaml
```

当前真实数据构建该 split 时，S32/S33/S34 合计 11015 条样本，切分为 train 8839、validation 1088、test 1088；每个 split 的 scene 组成和 scaler 来源会写入 `final_config.yaml` runtime metadata。

### 2604.05668 BEV-Fusion 本体复现

`configs/fusion/experiments/bev_fusion_2604/` 提供 arXiv:2604.05668 BEV-Fusion 本体复现配置族。`paper_full.yaml` 对齐论文主配置：DeepSense6G S32/S33/S34、5 帧历史、`future_beam1`、64 beam、image/radar/GPS/LiDAR 四模态、128x128 BEV、`d_model=256`、camera-to-BEV 3 层 4 heads、temporal transformer 4 层 4 heads、focal loss `gamma=2`、AdamW `lr=1e-4, weight_decay=1e-2` 和 `2604_linear_topk` metric profile。`low_memory.yaml` 与 `smoke.yaml` 只用于工程验证，会在 runtime metadata 中标记 `paper_approximation: true` 或 `mock_data: true`。

GPS spatial pathway 使用 sequence CSV 里的历史 `gps*` 与 `bs_gps*` 路径在线生成未标准化 UE-minus-BS relative XY，作为 `gps_bev_xy_batch` 传入模型；`gps_batch` 仍可按训练 split 使用 StandardScaler。full 配置默认 GPS BEV ROI 为 `[-80, 80, -80, 80]` 米，越界点裁剪并写入 diagnostics。horizontal flip 默认关闭；在 beam index reversal 有单元测试前，不应把它加入 paper full 训练。

推荐先跑 smoke 和配置/forward 测试，再准备真实 LiDAR BEV cache 并启动 full/low-memory：

```bash
conda run -n kd_mm_beam pytest tests/test_bev_fusion_2604.py -q
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/low_memory.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/paper_full.yaml
```

Ablation 配置位于 `configs/fusion/experiments/bev_fusion_2604/ablations/`，覆盖 `without_camera`、`without_lidar`、`without_radar`、`without_gps`、`one_d_fusion`、`single_frame`、`mean_pool_temporal`、`gps_spatial_only` 和 `gps_global_only`。这些配置继承同一 S32/S33/S34 split 和 linear DBA 口径，结果应按 `ablation_name` 分组展示：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/ablations/without_camera.yaml
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/experiments/bev_fusion_2604/ablations/gps_global_only.yaml
```

训练和评估产物写入 ignored 的 `outputs/scenegroup_s32_s34/<run_name>/` 或 `outputs/evaluations/`。报告 helper `build_bev_fusion_2604_report()` 使用 `linear_dba`、Top-K、`macro_linear_dba`、`weighted_overall_linear_dba`、paper target、split/seed/sample count、mock/real 标记、参数量和本机硬件/latency 字段；没有作者 exact split、seed、代码和权重时，报告必须保留 `paper_exact_split_available: false`，不得声称严格复现论文 exact result 或 H100 latency。

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

训练输出默认写入 `outputs/scene<id>/<run_name>/` 或 `outputs/scenegroup_<range-or-list>/<run_name>/`，包括 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`metrics.json`、checkpoint、TensorBoard 和可选 normalization/target artifact。评估集合默认写入 `outputs/evaluations/<study_id>/`，长期分析写入 `outputs/analysis/` 或当前保留的 `outputs/visual_analysis/`，可再生成 cache 写入 `outputs/cache/`，registry 写入当前 scene/scenegroup 下的 `best_checkpoints/`。根级 `outputs/<run_name>/`、数字场景根、根级 `outputs/best_checkpoints/` 和 `outputs/eval_*` 只作为 legacy/archive 审计对象。使用 virtual/overlay config 时，运行产物仍保存完整解析配置，不依赖原始 YAML 文件继续存在。
