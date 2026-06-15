# KD for Sensing

本仓库提供基于 `src/kd_sensing` 包的多模态少样本跨场景 beam prediction 工作流，当前主线收敛到 Image+GPS JEPA query-pool、JEPA-MSAC Scenario 32 workflow、paired baseline/control、vision-position baseline suite、Arnold22 Camera AE+GPS Direct、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、JEPA visual analysis、预处理和 manifest 诊断导出入口。

蒸馏训练、HiST-Beam、GPS coarse anchor、Top8 selector、GPS residual 和 camera residual 研究线已经退役。当前 quickstart、BGAM、GPS v2 和 calibration workflow 都只构建单个 `model.primary` 主模型；旧 `teacher_no_kd`、`student_no_kd`、`no_kd`、`logits_kd`、`rkd`、`distillation.*`、`configs/hist_beam/*`、`hist_beam_fusion` 和 `kd-sensing-hist-beam-loso` 会被 migration guard 或 registry 拒绝，并提示使用当前入口。历史输出和权重只作为只读复现资料保留。

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
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help
conda run -n kd_mm_beam kd-sensing-visualize-modalities --help
conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help
conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help
conda run -n kd_mm_beam kd-sensing-run-jepa-msac --help
conda run -n kd_mm_beam kd-sensing-run-deepsense6g-gps-lidar-bgam --help
conda run -n kd_mm_beam kd-sensing-run-mmw-town-gps-lidar-bgam --help
conda run -n kd_mm_beam kd-sensing-run-amr-net-gps-image --help
```

等价包内 CLI 入口形如：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.train --help
conda run -n kd_mm_beam python -m kd_sensing.cli.export_viewer_manifest --help
```

## 目录概览

```text
configs/          # 训练、评估和预处理配置；高级 fusion 优先由 canonical/overlay recipe 生成
docs/             # 主线模型目录、协议表、结果账本、实验矩阵、扩展指南和性能调优说明
openspec/specs/   # 当前需求和架构契约
scripts/          # 保留的薄 alias、研究诊断和数据准备脚本
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

触碰训练、数据集、诊断、CLI、配置解析或模型 forward 时，追加对应 focused tests；例如配置加载和 manifest/JEPA visual analysis/JEPA-MSAC 相关改动可运行：

```bash
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_jepa_visual_analysis.py -q
conda run -n kd_mm_beam pytest tests/test_jepa_msac.py -q
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

`fraction` 和 `num_samples` 二选一；`seed` 为空时默认使用 `experiment.seed`。默认 `rotate_each_epoch=true`，会按绝对 epoch 轮换无放回抽样，resume 后同一 epoch 的样本选择仍可复现；设置 `rotate_each_epoch=false` 可固定同一小子集用于排障。运行产物会在 `train_log.json`、`final_config.yaml` 的 runtime metadata 中记录完整 train 样本数、每 epoch 有效样本数、seed、轮换设置和是否退化为完整 epoch。更完整的吞吐和 cache 说明见 [docs/training_throughput.md](docs/training_throughput.md)。

评估：

```bash
conda run -n kd_mm_beam kd-sensing-evaluate \
  --config configs/image/strong.yaml \
  --weights outputs/scene31/image_strong/checkpoints/best.pth
```

已退役入口：

HiST-Beam LOSO、history-anchor Hist、P3/V7/V8/V9 Hist probe、image-only legal crossroad probe、GPS coarse anchor、Top8 selector、GPS residual、camera residual 和 Raymobtime s008 不再作为当前可运行入口维护。旧 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、`hist_beam_fusion`、`configs/raymobtime/*`、`configs/preprocess/raymobtime_s008_*.yaml` 和 `raymobtime_s008` dataset/model/preprocessor 名称会快速失败并说明研究线已退役；当前跨场景 follow-up 使用后文的 DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening 和 viewer workflow。

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

Viewer manifest 导出：

```bash
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/cache/diagnostics/viewer_manifest \
  --scenes 9,32
```

`kd-sensing-visualize-modalities` 保留为包内薄 alias，只委托 manifest 导出 CLI，不恢复旧静态 PNG 总览图流程。推荐命令仍是 `kd-sensing-export-viewer-manifest` 或 `python -m kd_sensing.cli.export_viewer_manifest`。

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

Benchmark 产物写入 ignored 的 `outputs/analysis/...`，包括 `benchmark_manifest.json`、`tables/metrics_by_condition.csv`、`tables/robustness_summary.csv`、`tables/shortcut_reliance_summary.csv` 和可选曲线图。真实 BeamBench-fair 矩阵使用 `configs/diagnostics/jepa_gps_shortcut_benchmark_beambench_fair.yaml`，其中 checkpoint 路径是本地占位，需要替换为实际 run；不要提交真实 checkpoint、metrics、figures、cache 或 reports。`kd-sensing-jepa-visual-analysis` 可通过 `benchmark.runner_manifest=<path>` 只读消费 runner manifest，生成 `benchmark_robustness_matrix.csv`、GPS collapse/image degradation/temporal delay 曲线和 GPS shortcut reliance 报告段落。

JEPA-MSAC Scenario 32 workflow：

```bash
conda run -n kd_mm_beam kd-sensing-run-jepa-msac \
  --config configs/pretraining/jepa_msac_s32_smoke.yaml \
  --stage report \
  --dry-run
```

该入口复现 arXiv:2603.29796 的两阶段 workflow 边界：Stage 1 temporal block-masked JEPA，Stage 2 frozen backbone localization/beam/RSSI heads，报告 ADE/FDE、Top-1/3、L1-RSRP diff、RSSI RMSE/MAE、RRankMe/RLDA schema。Smoke 不读取真实 `dataset/`；paper-aligned 配置 `configs/pretraining/jepa_msac_s32_paper.yaml` 需要本地 Scenario 32 字段审计通过。RF 只作为 workflow-local beam-power history 映射，不是新的 canonical modality。未完成长训练前 claim status 保持 `unverified`、`local-ready`、`blocked` 或 `mock/smoke`。

模态 difficulty profile 用于描述输入难度条件，不是新模态，也不会新增 `delayed_gps`、`image_hard` 等模型输入分支。profile 复用 canonical modality key，例如 `gps` 和 `image`，只扰动输入 tensor 及 `gps_valid_mask`、`gps_source_index`、`image_degradation_metadata` 等输入可靠性 metadata；`target_beam`、`beam_power`、soft target、sample id 和 split metadata 会被 guard 保护。示例配置位于 `configs/difficulty/`，覆盖 clean baseline、GPS mild async training、GPS severe async evaluation、GPS/image dropout training 和 image hard degradation evaluation sweep。新增 operator 时，在 `kd_sensing.data.difficulty.operators` 中实现轻量 batch transform，并通过 `DIFFICULTY_OPERATORS` 显式注册；训练、评估和 benchmark 会复用同一 profile/schema/pipeline。启用 difficulty 后，resolved profile、digest、stage/split、seed、warnings 和 replay metadata 会写入 `final_config.yaml`、runtime metadata 或 benchmark manifest；表格、图、cache 和 debug 输出仍只写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定目录。

AMR-Net_gps_image source audit：

```bash
conda run -n kd_mm_beam kd-sensing-run-amr-net-gps-image \
  --config configs/baselines/amr_net_gps_image.yaml
```

该入口默认只生成 AMR-Net_gps_image source-audit report 和 deterministic mock metrics，输出到 ignored 的 `outputs/analysis/amr_net_gps_image/`。当前公开 metadata 显示 IEEE document `11282996` 与 DeepSense6G Scenario 23 作者代码对应的 IEEE document `10000718` 不一致，因此 claim status 默认为 `blocked_official` / `mock_smoke`；不得把 Scenario 23 local substitute 写成 official reproduction，也不得启用 LiDAR。

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

### MMW Town GPS v2 logits for BGAM

MMW 是 GPS pseudo-history BGAM 的第一阶段主数据集，用来对照 arXiv:2603.15093v1。BGAM 默认使用 `mapping_enabled`、64-beam circular label space 和 MMW GPS v2 frozen logits；候选 manifest 由 BGAM manifest 准备流程消费或生成，不再提供 standalone Top8 manifest CLI/config。先重跑 GPS v2 并保存 logits：

```bash
conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 \
  --config configs/mmw_town_gps_adapter_v2.yaml \
  --label-space mapping_enabled \
  --save-logits \
  --save-prior-probs
```

BGAM manifest builder 从 `gps_logits.npy` 重新计算候选，不从 `predictions.csv` 的 Top5 字段截断推导。MMW calibration 是 scene-specific，所以 logits index、predictions、support manifest 和 BGAM candidate manifest 会按 scene 校验 `beam_label_mapping_fingerprint`。若 future beam power path 可用，manifest 还会写 `gps_normalized_gain`、candidate normalized gain 和 oracle normalized gain，便于和论文的 normalized gain 指标对照。

### MMW Town GPS+LiDAR BGAM Reranker

MMW BGAM 默认配置是 `configs/mmw_town_gps_lidar_bgam.yaml`，输出写入 `outputs/analysis/mmw_town_gps_lidar_bgam/mapping_enabled/`。workflow 复用 MMW GPS v2 Top8 candidates，把 GPS v2 作为 frozen prior，并用历史 GPS pseudo label 引导 RSU/BS-side LiDAR BEV/raw point cloud 在 GPS Top8 内重排候选 beam。`mapping_disabled` 只作为显式 raw-label 对照。

```bash
conda run -n kd_mm_beam kd-sensing-prepare-mmw-town-gps-lidar-bgam-manifest \
  --config configs/mmw_town_gps_lidar_bgam.yaml \
  --label-space mapping_enabled \
  --topk 8

conda run -n kd_mm_beam kd-sensing-run-mmw-town-gps-lidar-bgam \
  --config configs/mmw_town_gps_lidar_bgam.yaml \
  --label-space mapping_enabled \
  --topk 8

conda run -n kd_mm_beam kd-sensing-evaluate-mmw-town-gps-lidar-bgam \
  --config configs/mmw_town_gps_lidar_bgam.yaml \
  --ckpt outputs/analysis/mmw_town_gps_lidar_bgam/mapping_enabled/checkpoints/gps_pseudo_history_soft_bgam.pt \
  --output-dir outputs/analysis/mmw_town_gps_lidar_bgam/eval_smoke
```

pseudo-history 默认按 `scene + agent + split` 分组，按 nearest-past 构造 `history_pseudo_beams`、prob、entropy、valid mask 和 timestamps，避免不同车辆或 train/test split 串历史。LiDAR 默认优先使用 frame manifest 中的 RSU LiDAR path；BEV cache 缺失时可按配置从 raw `.pcd` 重建。future ground-truth beam 只用于 loss/evaluation/report，不用于 BGAM mask、pseudo-history、normalizer fit 或 checkpoint selection。

默认 ablation 包含 `gps_only`、`lidar_only_no_bgam`、`gps_lidar_no_bgam`、`gps_lidar_topk_union_bgam`、`gps_pseudo_history_soft_bgam`、`gps_pseudo_history_topk_union_bgam` 和 `gps_pseudo_history_per_candidate_rerank`。结果先看 `summary_overall.csv`、`summary_by_scene.csv`、`summary_by_bgam_mode.csv`、`predictions.csv` 和 `manifest/pseudo_history_summary.csv`，比较 TopK、DBA、mean/median circular error、pseudo-history coverage/entropy、normalized gain 和 delta vs GPS。若启用 `oracle_history_bgam_upper_bound`，summary 会标记为上界，默认 best ablation 不会从 oracle 中选择。

MMW Town10 本地 zip 默认放在 `dataset/_downloads/MMW/<condition>/Sensor_Data` 和 `dataset/_downloads/MMW/<condition>/Channel_Data`，prepared 产物写入 `dataset/MMW/<condition>/Prepared/<scenario>`。准备流程只解压必要的 sensor zip 和共用的 `Town10.zip` channel 包，不移动或删除下载文件；已下载但暂不处理的场景会在 availability 中保持 `pending` 或 `downloaded_unprepared`。

```bash
conda run -n kd_mm_beam python scripts/mmw/prepare_town10_skybridge.py \
  --config configs/preprocess/mmw_town10_skybridge.yaml
```

MMW sequence split 默认使用 `group_safe_time_block` 协议，按连续片段、agent 和 time block 切分，并写出 `split_metadata.json`。metadata 会记录 `split_protocol=mmw_sequence_split_v2`、`split_strategy`、guard band、train/test group、样本数、标签分布、泄漏诊断和 `strict_validation_eligible`；训练、评估和 quick summary 会消费这些字段。旧随机滑窗切分不再作为公开准备或 split builder 协议支持；已有旧 CSV 应使用新的 split tag 重新生成 group-safe split，缺失 metadata 或 `strict_validation_eligible=false` 的产物会被保守标记为不进入 strict 主结论。

MMW beam label calibration 默认关闭，普通训练和评估沿用 raw 64-beam label space。需要按 GPS-angle 诊断重映射 label 时，在当前 MMW GPS v2 或 BGAM 配置中显式选择 `mapping_enabled` 或设置 `data.dataset.beam_label_calibration.enabled=true` 及 offset/mapping file。启用后 `input_beam`、`target_beam`、soft label、beamspace physical label、prediction/diagnostic metadata 会声明 `beam_label_space` 和 mapping fingerprint；`mmwave` sensing power vector 仍保持原始顺序。raw-label 旧 checkpoint 和 mapped-label 新 run 不应直接混比，除非报告明确执行 inverse mapping 或按 label space 分组。

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

DeepSense6G GPS residual fusion、camera residual、GPS coarse anchor 和 Top8 selector 训练路线已经退役；对应配置、console scripts、engine/model/loss 和 focused tests 不再维护。BGAM 模块仍保留，继续使用 GPS v2 logits/candidate manifest 作为当前 reranker workflow 的输入。

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

每次准备完成后会写 `dataset/MMW/<condition>/data_availability.json` 和 `dataset/MMW/data_availability.json`。当前 MMW 跨场景验证不再生成 HiST scenario-LOSO 计划；推荐顺序是先完成 Town10 数据准备和 group-safe split，再运行 MMW GPS v2 和 GPS+LiDAR BGAM。旧 P3/V7/V8/V9 本地输出如果仍在 `outputs/` 中，只作为历史分析资料，不再作为 README 当前命令来源。

包含 image 或 LiDAR 的长跑通常先受 CPU image 解码、DataLoader wait、cache coverage 和 worker RSS 限制。长跑前建议使用 [docs/training_throughput.md](docs/training_throughput.md) 中的 profile 与并行推荐流程；推荐器会优先给出 `num_workers`、`batch_size`、并行度、`persistent_workers` 和 `output.progress.enabled=false` 的保守覆盖。启用 RGB/ImageNet 派生缓存时使用 `data.cache.image.policy=auto|read_only|rebuild|off`，可预热：

```bash
conda run -n kd_mm_beam python scripts/preprocess.py \
  --config configs/preprocess/mmw_image_derived_cache.yaml
```

若 profile 或日志出现 loader wait 支配 step、退出码 137、`Killed` 或 worker RSS 过高，优先降低并行 runs、batch size 和 train workers，关闭 persistent workers，或预热 image-derived cache；不要默认继续增加 worker。

## Viewer Manifest

仓库级 Gradio viewer 支持已退役；当前保留包内 manifest 导出能力，供外部查看器、离线诊断或 JEPA visual analysis 消费。`kd-sensing-visualize-modalities` 仍是薄 alias，只委托 `kd-sensing-export-viewer-manifest`，不恢复旧静态 PNG 总览图或仓库级 Web UI。

离线 manifest 导出推荐：

```bash
conda run -n kd_mm_beam kd-sensing-export-viewer-manifest \
  --config configs/diagnostics/modality_visualization.yaml \
  --cache-dir outputs/cache/diagnostics/viewer_manifest \
  --scenes 32 \
  --predictions outputs/eval/predictions.json \
  --quality outputs/eval/quality.json \
  --gate outputs/eval/gate.json
```

## 文档索引

- AI/维护者修改前导航：[docs/agent_navigation.md](docs/agent_navigation.md)
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

旧的顶层脚本入口已移除；请使用 console script、包内 CLI 或保留的 `scripts/` 薄 alias。

| 旧命令 | 当前入口 |
| --- | --- |
| `python train_image.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/image/<mode>.yaml ...` |
| `python train_both.py ...` | `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/<mode>.yaml ...` |
| `python test_model_image.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/image/<mode>.yaml --weights <path>` |
| `python test_model_both.py ...` | `conda run -n kd_mm_beam kd-sensing-evaluate --config configs/fusion/<mode>.yaml --weights <path>` |
| `python CSV_process.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/radar_ra.yaml` |
| `python gen_data_seq.py ...` | `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra.yaml` |
