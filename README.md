# KD for Sensing

本仓库提供基于 `src/kd_sensing` 包的多模态少样本跨场景 beam prediction 工作流，当前主线收敛到 Image+GPS JEPA query-pool、paired baseline/control、vision-position baseline suite、Arnold22 Camera AE+GPS Direct、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理和保留诊断入口。

蒸馏训练、HiST-Beam、GPS coarse anchor、Top8 selector、GPS residual、camera residual、DeepSense6G/MMW BGAM、viewer manifest、仓库级 Gradio viewer、AMR-Net_gps_image mock/source-audit runner 和 JEPA-MSAC Scenario 32 mock workflow 已经退役。当前 quickstart、GPS v2 和 calibration workflow 都只构建单个 `model.primary` 主模型或明确的诊断 workflow；旧 `teacher_no_kd`、`student_no_kd`、`no_kd`、`logits_kd`、`rkd`、`distillation.*`、`configs/hist_beam/*`、`hist_beam_fusion`、BGAM 配置、viewer manifest 配置、AMR/JEPA-MSAC 配置和对应 `kd-sensing-*` 命令会被 migration guard 或入口清单拒绝。历史输出和权重只作为只读复现资料保留。

## 安装

所有项目相关 Python 命令都使用 `kd_mm_beam` 环境：

```bash
conda run -n kd_mm_beam python -m pip install -e .
conda run -n kd_mm_beam python -c "import kd_sensing"
```

安装后可用 console scripts：

```bash
conda run -n kd_mm_beam kd-sensing-train --help
conda run -n kd_mm_beam kd-sensing-evaluate --help
conda run -n kd_mm_beam kd-sensing-preprocess --help
conda run -n kd_mm_beam kd-sensing-runs --help
conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --help
conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help
```

等价包内 CLI 入口形如：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.train --help
conda run -n kd_mm_beam python -m kd_sensing.cli.jepa_visual_analysis --help
```

## 目录概览

```text
configs/          # 训练、评估和预处理配置；高级 fusion 优先由 canonical/overlay recipe 生成
docs/             # 主线模型目录、协议表、结果账本、实验矩阵、扩展指南和性能调优说明
openspec/specs/   # 当前需求和架构契约
scripts/          # 保留的研究诊断、数据准备和 shell orchestration 脚本
src/kd_sensing/   # 包内 CLI、config、data、engine、models、diagnostics 等实现
tests/            # 架构边界、配置加载、训练/诊断单元测试
tools/analysis/   # 研究分析脚本
```

配置相对路径从项目根目录解析，因此可以在子目录中启动命令。

## 快速健康检查

窄改动优先运行相关测试。涉及 OpenSpec、架构、导入边界、CLI 或公共 workflow 时，按层运行：

```bash
openspec validate strengthen-project-health-guardrails --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q
```

触碰训练、数据集、诊断、CLI、配置解析或模型 forward 时，追加对应 focused tests；例如配置加载和 manifest/JEPA visual analysis 相关改动可运行：

```bash
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_visual_analysis.py -q
```

这些检查不启动真实训练、不读取 `dataset/` 真实数据、不写入 checkpoint 或训练输出。最终回归：

```bash
conda run -n kd_mm_beam pytest -q
```

## 主要入口

训练：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml
```

快速调试训练时，可以启用 train epoch 子采样，只减少每个 epoch 的训练 step，不改 train CSV、不缩小 validation/test split，也不替代 `data.dataset.portion`：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_lightweight.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.fraction=0.1 \
  -o output.progress.enabled=false
```

也可以用固定样本数限制每个 epoch：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/lightweight.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.num_samples=256
```

`configs/<image|radar|gps|lidar|mmwave>/{strong,lightweight,supervised}.yaml` 保留熟悉的文件名和 run name，但普通 baseline 的主模型已经统一为 `model.primary.type: modular_sequence`；旧 `*_strong` / `*_lightweight` registry 名只保留 removed guard 和迁移提示。

`fraction` 和 `num_samples` 二选一；`seed` 为空时默认使用 `experiment.seed`。默认 `rotate_each_epoch=true`，会按绝对 epoch 轮换无放回抽样，resume 后同一 epoch 的样本选择仍可复现；设置 `rotate_each_epoch=false` 可固定同一小子集用于排障。运行产物会在 `train_log.json`、`final_config.yaml` 的 runtime metadata 中记录完整 train 样本数、每 epoch 有效样本数、seed、轮换设置和是否退化为完整 epoch。更完整的吞吐和 cache 说明见 [docs/training_throughput.md](docs/training_throughput.md)。

评估：

```bash
conda run -n kd_mm_beam kd-sensing-evaluate \
  --config configs/image/strong.yaml \
  --weights outputs/scene31/image_strong/checkpoints/best.pth
```

已退役入口：

HiST-Beam LOSO、history-anchor Hist、P3/V7/V8/V9 Hist probe、image-only legal crossroad probe、GPS coarse anchor、Top8 selector、GPS residual、camera residual、Raymobtime s008、BGAM、viewer manifest、仓库级 Gradio viewer、AMR-Net_gps_image mock/source-audit runner 和 JEPA-MSAC Scenario 32 mock workflow 不再作为当前可运行入口维护。旧 `kd-sensing-hist-beam-loso`、`kd-sensing-run-amr-net-gps-image`、`kd-sensing-run-jepa-msac`、`configs/hist_beam/*`、`configs/baselines/amr_net_gps_image.yaml`、`configs/pretraining/jepa_msac_s32_*.yaml`、`hist_beam_fusion`、`configs/raymobtime/*`、`configs/preprocess/raymobtime_s008_*.yaml`、`configs/*bgam*.yaml`、`configs/diagnostics/modality_visualization.yaml` 和相关 dataset/model/preprocessor 名称会快速失败并说明研究线已退役；当前跨场景 follow-up 使用后文的 MMW GPS v2、CSI hardening、JEPA visual analysis 和 GPS shortcut benchmark。

实验运行索引：

```bash
conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs
conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs --format json \
  --state running --state killed --output outputs/analysis/run_index.json
```

`kd-sensing-runs` 只读扫描本地 `outputs/`、`logs/`、当前 Python 进程和可用资源快照，不删除、不移动、不重写训练产物、日志、checkpoint、cache 或 TensorBoard 文件。状态分类包括 `running`、`complete`、`started_no_metrics`、`partial`、`failed`、`killed`、`waiting`、`stale` 和 `unknown`；JSON 输出稳定包含 `generated_at`、`roots`、`runs`、`resources` 和 `warnings`。
默认扫描 `outputs/` 时会跳过 `outputs/cache/`、`outputs/archive/` 和 `outputs/cleanup_manifests/` 等非当前 run 分区；如需审计这些目录，可显式传入 `--outputs outputs/cache` 或 `--outputs outputs/archive`。

本地产物清理 manifest：

```bash
conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --root outputs --root logs --root .pytest_cache
conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --delete \
  --manifest outputs/cleanup_manifests/runtime_cleanup_<timestamp>.json \
  --confirm-delete
```

`kd-sensing-clean-runtime-artifacts` 默认只生成 dry-run JSON manifest，不删除、不移动、不压缩、不重写本地数据、输出、日志、cache、checkpoint、源码、配置、文档或 OpenSpec artifact。删除阶段必须显式传入 manifest 和 `--confirm-delete`，并在执行前重新验证路径仍未被 git 跟踪、未受保护且仍位于 manifest 扫描根内。

本地 outputs 整理 manifest：

```bash
conda run -n kd_mm_beam kd-sensing-organize-runtime-outputs --outputs-root outputs
conda run -n kd_mm_beam kd-sensing-organize-runtime-outputs --execute \
  --manifest outputs/cleanup_manifests/runtime_organize_<timestamp>.json \
  --confirm-organize
```

`kd-sensing-organize-runtime-outputs` 的默认模式只生成 move/archive/protect/review 计划，不移动、不删除、不重写任何本地产物。执行阶段必须显式确认，并会重新检查 source 状态、git tracked 保护和 target 冲突。

预处理：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra_gps_lidar.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_snapshot_next_frame.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
```

Viewer manifest 和 `kd-sensing-visualize-modalities` alias 已退役，不再作为包内诊断入口。需要论文图、case payload 或 benchmark 证据时使用下面的 JEPA visual analysis、GPS shortcut benchmark 或其它明确 current 的诊断入口。

JEPA visual analysis 离线论文图导出：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis \
  --analysis-config configs/diagnostics/jepa_visual_analysis_2604.yaml \
  --output-dir outputs/visual_analysis/jepa_query_pool_2604 \
  --force
```

该入口默认只读模型 config、checkpoint、split 和已有 cache；新增产物只写入 `--output-dir` 下的 `figures/`、`tables/`、`cache/`、`case_payloads/`、`analysis_manifest.json` 和 `report.md`。示例配置中的 `fair_base`、`fair_gps_biased` 可按本地 checkpoint 路径替换；没有 attention 或 UMAP 时分析会在 manifest/report 中记录 warning 并降级到剩余图表。解释结论时优先引用 `report.md`、`tables/comparison_samples.csv`、`tables/embedding_neighbors.csv` 和对应 PNG/SVG 图，同时保留 caveat：投影和 attention 是诊断证据，不是单独的因果证明。

JEPA vs GPS shortcut benchmark：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark \
  --manifest configs/diagnostics/jepa_gps_shortcut_benchmark_smoke.yaml \
  --output-dir outputs/analysis/jepa_gps_shortcut_benchmark/smoke \
  --force
```

Benchmark 产物写入 ignored 的 `outputs/analysis/...`，包括 `benchmark_manifest.json`、`tables/metrics_by_condition.csv`、`tables/robustness_summary.csv`、`tables/shortcut_reliance_summary.csv` 和可选曲线图。Scenario D / CxD smoke 使用 `configs/diagnostics/jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml`，在保留 `results/scenario_d_image_observability.csv`、`results/heatmap_cx_dy.npy` 的同时写出 `results/cxd_phase_diagram.csv`、`results/cxd_phase_heatmap.npy`、`results/modality_dominance.csv`、`results/crossing_region_Cx_Dy.json`、`results/failure_mode_decomposition.csv` 和对应 PNG；synthetic dominance 行只标记 mock/unavailable，不冒充真实解释证据。真实 BeamBench-fair 矩阵使用 `configs/diagnostics/jepa_gps_shortcut_benchmark_beambench_fair.yaml`，其中 checkpoint 路径是本地占位，需要替换为实际 run；不要提交真实 checkpoint、metrics、figures、cache 或 reports。`kd-sensing-jepa-visual-analysis` 可通过 `benchmark.runner_manifest=<path>` 只读消费 runner manifest，生成 `benchmark_robustness_matrix.csv`、GPS collapse/image degradation/temporal delay 曲线和 GPS shortcut reliance 报告段落。

JEPA-MSAC Scenario 32 workflow 已退役为 tombstone，不再提供 current CLI、config、model、loss 或 objective。历史背景只说明它曾用于 arXiv:2603.29796 两阶段 workflow 审计；当前 JEPA 相关工作请使用 GPS-conditioned JEPA、JEPA visual analysis 或 GPS shortcut benchmark。

模态 difficulty profile 用于描述输入难度条件，不是新模态，也不会新增 `delayed_gps`、`image_hard` 等模型输入分支。profile 复用 canonical modality key，例如 `gps` 和 `image`，只扰动输入 tensor 及 `gps_valid_mask`、`gps_source_index`、`image_degradation_metadata` 等输入可靠性 metadata；`target_beam`、`beam_power`、soft target、sample id 和 split metadata 会被 guard 保护。示例配置位于 `configs/difficulty/`，覆盖 clean baseline、GPS mild async training、GPS severe async evaluation、GPS/image dropout training 和 image hard degradation evaluation sweep。新增 operator 时，在 `kd_sensing.data.difficulty.operators` 中实现轻量 batch transform，并通过 `DIFFICULTY_OPERATORS` 显式注册；训练、评估和 benchmark 会复用同一 profile/schema/pipeline。启用 difficulty 后，resolved profile、digest、stage/split、seed、warnings 和 replay metadata 会写入 `final_config.yaml`、runtime metadata 或 benchmark manifest；表格、图、cache 和 debug 输出仍只写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定目录。

AMR-Net_gps_image / IEEE `11282996` source-audit runner 已退役为 tombstone，不再提供 current CLI、config 或 mock metrics。保留的历史 caveat 是公开 metadata 与 DeepSense6G Scenario 23 作者包 document id 不一致，旧本地产物不能声明 official reproduction；当前 GPS+Image 对照使用 Vision-Position suite 或 BeamBench Image AE+GPS Direct。

## 配置和实验矩阵

单模态 canonical 配置使用：

- `configs/<image|radar|gps|lidar|mmwave>/strong.yaml`
- `configs/<image|radar|gps|lidar|mmwave>/lightweight.yaml`
- `configs/<image|radar|gps|lidar|mmwave>/supervised.yaml`

Fusion canonical 配置按固定模态顺序 `image -> radar -> gps -> lidar -> mmwave` 解析，命名为：

```text
configs/fusion/<canonical_slug>_<strong|lightweight>.yaml
```

旧 `teacher_no_kd` / `student_no_kd` / `no_kd` / `logits_kd` / `rkd` 路径不再作为支持入口存在；配置加载器会拒绝这些路径并给出迁移建议。不存在实体 YAML 的 `configs/fusion/<slug>_logits_kd.yaml` 和 `configs/fusion/<slug>_rkd.yaml` 也不会由 virtual config 生成。

很多 fusion 路径是 virtual config：磁盘上没有实体 YAML 时，配置加载器会按 strong/lightweight canonical、snapshot、objective-aware 或当前保留的 overlay recipe 生成完整配置；实体 YAML 仍优先于生成规则。训练产物中的 `final_config.yaml` 和 `resolved_config.yaml` 保存完整解析结果。已退役研究线和 fusion KD alias 的旧配置路径不会被 virtual alias 接管。

当前主线横向说明分三层维护：模型目录见 [docs/mainline_model_catalog.md](docs/mainline_model_catalog.md)，参数协议见 [docs/experiment_protocols.md](docs/experiment_protocols.md)，可引用结果和 blocked 状态见 [docs/result_claims_registry.md](docs/result_claims_registry.md)。[docs/experiment_matrix.md](docs/experiment_matrix.md) 只保留 quickstart 顺序和关键 caveat；CSI hardening、snapshot next-frame、objective-aware fusion、MMW 和推荐实验顺序从这里跳转。

## 数据和产物边界

- `dataset/` 是本地数据输入，默认不提交；源码中只保留 `dataset/.gitkeep`。
- `outputs/`、`outputs/cache/`、`logs/`、legacy 根 `cache/`、TensorBoard 产物和新生成 checkpoint 是本地运行产物，默认不提交；新可再生成 cache 默认写入 `outputs/cache/`。
- 当前 runtime output 分区为：`outputs/cache/`、`outputs/cleanup_manifests/`、`outputs/analysis/`、`outputs/visual_analysis/`、`outputs/evaluations/`、`outputs/scene<id>/`、`outputs/scenegroup_*/` 和 `outputs/archive/`。新训练默认写入 scene/scenegroup scope；根级 `outputs/<run_name>/`、数字场景根和根级 `outputs/best_checkpoints/` 只按 legacy 输入审计。
- `All_models/` 中已跟踪权重是历史复现实验资料；新生成的 `.pth`、`.pt`、`.ckpt` 不应进入源码变更。
- 本地产物清理必须先生成 manifest；真正删除需要显式确认，且默认保护 `dataset/`、`All_models/`、源码、配置、文档、OpenSpec、已跟踪文件和活跃运行。
- 当前训练入口不读取蒸馏权重；评估入口仍可通过 `--weights` 指定待评估模型权重。

DeepSense6G 默认场景是 Scenario 31，数据根目录解析为 `dataset/DeepSense6G/scenario31`，单场景输出默认写入 `outputs/scene31/<run_name>/`；包含多个 `train_scenes`/`eval_scenes` 的配置默认写入 `outputs/scenegroup_<range-or-list>/<run_name>/`。可通过配置或 CLI override 切换场景：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/mmwave/strong.yaml data.dataset.scene=9
conda run -n kd_mm_beam kd-sensing-train --config configs/mmwave/strong.yaml data.dataset.scene=32
```

### MMW Town GPS-only v2 inputs and diagnostics

MMW 当前保留面聚焦 GPS-only v2、group-safe split、label-space calibration 和诊断图表。需要保存 logits/probs 供本地分析时可显式开启输出，但这些文件不再作为 BGAM 或 Top8 candidate manifest 的中间产物：

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 \
  --config configs/mmw_town_gps_adapter_v2.yaml \
  --label-space mapping_enabled \
  --save-logits \
  --save-prior-probs
```

DeepSense6G/MMW GPS+LiDAR BGAM、GPS pseudo-history BGAM、BGAM-only TopK candidate manifest/dataset/loss 和 debug mask/report 已退役；`configs/*bgam*.yaml`、BGAM prepare/run/evaluate console scripts 和 viewer manifest 命令不再作为当前入口维护。

MMW Town10 本地 zip 默认放在 `dataset/_downloads/MMW/<condition>/Sensor_Data` 和 `dataset/_downloads/MMW/<condition>/Channel_Data`，prepared 产物写入 `dataset/MMW/<condition>/Prepared/<scenario>`。准备流程只解压必要的 sensor zip 和共用的 `Town10.zip` channel 包，不移动或删除下载文件；已下载但暂不处理的场景会在 availability 中保持 `pending` 或 `downloaded_unprepared`。

```bash
conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py \
  --config configs/preprocess/mmw_town10_skybridge.yaml
```

MMW sequence split 默认使用 `group_safe_time_block` 协议，按连续片段、agent 和 time block 切分，并写出 `split_metadata.json`。metadata 会记录 `split_protocol=mmw_sequence_split_v2`、`split_strategy`、guard band、train/test group、样本数、标签分布、泄漏诊断和 `strict_validation_eligible`；训练、评估和 quick summary 会消费这些字段。旧随机滑窗切分不再作为公开准备或 split builder 协议支持；已有旧 CSV 应使用新的 split tag 重新生成 group-safe split，缺失 metadata 或 `strict_validation_eligible=false` 的产物会被保守标记为不进入 strict 主结论。

MMW beam label calibration 默认关闭，普通训练和评估沿用 raw 64-beam label space。需要按 GPS-angle 诊断重映射 label 时，在当前 MMW GPS v2 配置中显式选择 `mapping_enabled` 或设置 `data.dataset.beam_label_calibration.enabled=true` 及 offset/mapping file。启用后 `input_beam`、`target_beam`、soft label、beamspace physical label、prediction/diagnostic metadata 会声明 `beam_label_space` 和 mapping fingerprint；`mmwave` sensing power vector 仍保持原始顺序。raw-label 旧 checkpoint 和 mapped-label 新 run 不应直接混比，除非报告明确执行 inverse mapping 或按 label space 分组。

### MMW Town GPS-only v2: circular scene adapter

`configs/mmw_town_gps_adapter_v2.yaml` 提供显式 opt-in 的 GPS-only v2 诊断 workflow，用来解释四个 sunny/Town10 scene 中普通跨场景 GPS 分类器失败的原因：beam label 是环形拓扑，0/63 相邻；不同 scene 存在朝向、beam shift、分支轨迹和 label imbalance；crossroad 与 Hroad 的残差往往不是单个 global shift 能解释。v2 默认使用 `mapping_enabled` 复用既有 calibration artifact，也支持 `mapping_disabled` raw 64-beam 对照，所有 summary 与 predictions 都按 label space 分目录并记录 mapping fingerprint。

v2 只使用 BS/RSU-centric GPS 几何特征，不把 raw latitude/longitude 放入模型 tensor。主指标使用 circular beam distance，`summary_by_scene.csv` 可按 `protocol`、`ablation` 和 `scene` 过滤，重点看 `DBA`、`DBA_zero_ratio`、`mean_circular_error`、`median_circular_error`、`exact_acc`、`pm1_acc`、`pm2_acc`、`pm4_acc`、`top1`、`top3`、`top5`。`within_scene_train` 只作为同场景上界或 sanity protocol，不作为跨场景泛化结论。

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 \
  --config configs/mmw_town_gps_adapter_v2.yaml \
  --label-space mapping_enabled
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 \
  --config configs/mmw_town_gps_adapter_v2.yaml \
  --label-space mapping_disabled
conda run -n kd_mm_beam kd-sensing-plot-mmw-town-gps-v2 \
  --results-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled
conda run -n kd_mm_beam kd-sensing-compare-mmw-town-gps-v2 \
  --previous-dir outputs/analysis/mmw_town_label_distribution \
  --new-dir outputs/analysis/mmw_town_gps_adapter_v2/mapping_enabled
```

adapter 消融包含 `adapter_v1`、`circular_affine`、`circular_affine_spline` 和 `branch_mixture_circular`，并保留 `backbone_only`、`geo_only`、`geo_plus_backbone`、`branch_mixture_circular_weighted` 对照。crossroad/Hroad 的结构性问题优先查看 `residual_by_theta_bin.csv`、`residual_by_branch.csv` 和 plotter 生成的 signed residual、branch visualization；curvyroad/skybridge 则关注 few-shot support 后是否保持旧 target_adapt_beambench 收益。本 workflow 不实现 camera/LiDAR/radar/mmWave 多模态 residual correction，也不改变现有 GPS v1 或 MMW calibration 默认行为。

### Retired GPS residual routes

DeepSense6G GPS residual fusion、camera residual、GPS coarse anchor、Top8 selector 和 BGAM 训练路线已经退役；对应配置、console scripts、engine/model/loss、candidate manifest 支撑和 focused tests 不再维护。

可用 override 增量处理其它 sunny 场景：

```bash
conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py \
  --config configs/preprocess/mmw_town10_skybridge.yaml \
  -o mmw.sensor_zip=dataset/_downloads/MMW/sunny/Sensor_Data/Town10_crossroad_seed24.zip \
  -o mmw.channel_zip=dataset/_downloads/MMW/sunny/Channel_Data/Town10.zip \
  -o mmw.scenario=Town10_crossroad_seed24
```

已有 frame manifest 时，可用公开 split builder 生成独立 strict split tag，避免复用旧 `l5p6` random-window CSV：

```bash
conda run -n kd_mm_beam python scripts/mmw/build_sequence_splits_from_manifest.py \
  --data-root dataset/MMW/sunny \
  --scene Town10_crossroad_seed24 \
  --seq-len 5 \
  --pred-len 6 \
  --split-tag l5p6_group_safe \
  --split-strategy group_safe_time_block
```

每次准备完成后会写 `dataset/MMW/<condition>/data_availability.json` 和 `dataset/MMW/data_availability.json`。当前 MMW 跨场景验证不再生成 HiST scenario-LOSO 计划；推荐顺序是先完成 Town10 数据准备和 group-safe split，再运行 MMW GPS v2、CSI hardening 或保留诊断。旧 P3/V7/V8/V9/BGAM 本地输出如果仍在 `outputs/` 中，只作为历史分析资料，不再作为 README 当前命令来源。

包含 image 或 LiDAR 的长跑通常先受 CPU image 解码、DataLoader wait、cache coverage 和 worker RSS 限制。长跑前建议使用 [docs/training_throughput.md](docs/training_throughput.md) 中的 profile 与并行推荐流程；推荐器会优先给出 `num_workers`、`batch_size`、并行度、`persistent_workers` 和 `output.progress.enabled=false` 的保守覆盖。启用 RGB/ImageNet 派生缓存时使用 `data.cache.image.policy=auto|read_only|rebuild|off`，可预热：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess \
  --config configs/preprocess/mmw_image_derived_cache.yaml
```

若 profile 或日志出现 loader wait 支配 step、退出码 137、`Killed` 或 worker RSS 过高，优先降低并行 runs、batch size 和 train workers，关闭 persistent workers，或预热 image-derived cache；不要默认继续增加 worker。

## Retired Viewer Manifest

仓库级 Gradio viewer、viewer manifest 导出、viewer prediction export 和 `kd-sensing-visualize-modalities` alias 已退役；不再提供兼容 stub、薄 alias 或 virtual config。当前诊断请使用 `kd-sensing-jepa-visual-analysis`、`kd-sensing-jepa-gps-shortcut-benchmark` 或其它明确 current 的诊断入口。

## 文档索引

- AI/维护者修改前导航：[docs/agent_navigation.md](docs/agent_navigation.md)
- 机器可读维护上下文索引：[docs/maintainer_context_index.yaml](docs/maintainer_context_index.yaml)
- 当前主线模型目录：[docs/mainline_model_catalog.md](docs/mainline_model_catalog.md)
- 实验协议和参数口径：[docs/experiment_protocols.md](docs/experiment_protocols.md)
- 结果和 claim 账本：[docs/result_claims_registry.md](docs/result_claims_registry.md)
- 实验矩阵 quickstart 和推荐运行顺序：[docs/experiment_matrix.md](docs/experiment_matrix.md)
- 研究结论和历史方案收束：[docs/research_notes.md](docs/research_notes.md)
- 训练吞吐、cache 和并行建议：[docs/training_throughput.md](docs/training_throughput.md)
- 新组件扩展指南：[docs/extension_guide.md](docs/extension_guide.md)
- 项目表面积 inventory：[docs/project_surface_inventory.md](docs/project_surface_inventory.md)
- 架构与需求契约：`openspec/specs/`

## 破坏性变更

旧的顶层脚本入口和 Python thin alias 已移除；请使用 console script、包内 CLI 或保留的研究/数据准备脚本。

| 旧命令 | 当前入口 |
| --- | --- |
| `python train_image.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/image/<mode>.yaml ...` |
| `python train_both.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/<mode>.yaml ...` |
| `python test_model_image.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/image/<mode>.yaml --weights <path>` |
| `python test_model_both.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/fusion/<mode>.yaml --weights <path>` |
| `python CSV_process.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/radar_ra.yaml` |
| `python gen_data_seq.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra.yaml` |
