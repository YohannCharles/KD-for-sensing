# 项目表面积 Inventory

本 inventory 记录 `refine-source-architecture-and-entry-surface` 的可审计基线。统计口径只覆盖源码、配置、文档和 OpenSpec artifact；`dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包和其它本地运行产物不属于本 change 的处理范围。

## 源码热点模块

本批次拆分的热点 facade 与职责模块如下：

- `tools/visualization/viewer_utils.py` 保留兼容导出；manifest 读取、路径解析和 scene/split/show mode 过滤迁移到 `tools/visualization/viewer_manifest_io.py`，图表构造迁移到 `tools/visualization/viewer_figures.py`，prediction summary 和 legacy prediction adapter 迁移到 `tools/visualization/viewer_prediction_tables.py`，常量迁移到 `tools/visualization/viewer_constants.py`。
- `src/kd_sensing/preprocessing/raymobtime_s008.py` 保留 preprocessor registry；paths/audit、index、beam labels、ray features 和 cache writer 分别迁移到 `raymobtime_s008_paths.py`、`raymobtime_s008_index.py`、`raymobtime_s008_beam_labels.py`、`raymobtime_s008_ray_features.py`、`raymobtime_s008_cache.py`，共享常量和窄 helper 在 `raymobtime_s008_common.py`。
- `src/kd_sensing/models/csi.py` 保留公开 import 路径；pilot estimation、CSI hardening、view tokenizer/fusion、debug helpers 和 encoder registry glue 分别迁移到 `csi_estimation.py`、`csi_hardening.py`、`csi_views.py`、`csi_debug.py`、`csi_encoder.py`。
- `src/kd_sensing/engine/objective_metadata.py` 保留公开兼容 facade；objective 名称、默认 metric、metric alias 和 mode 表迁移到 `src/kd_sensing/engine/objectives/registry.py`，history fields 与 TensorBoard scalar schema 迁移到 `src/kd_sensing/engine/objectives/history.py`，runtime metadata/validation helper 在 `src/kd_sensing/engine/objectives/metadata.py`。
- `src/kd_sensing/diagnostics/viewer_manifest.py` 保留 manifest 导出公开 orchestration；sample id/JSON schema、cache metadata、row path resolution、prediction/quality/gate merge 和 asset writer 分别迁移到 `src/kd_sensing/diagnostics/viewer_manifest_schema.py`、`src/kd_sensing/diagnostics/viewer_manifest_cache.py`、`src/kd_sensing/diagnostics/viewer_manifest_paths.py`、`src/kd_sensing/diagnostics/viewer_manifest_merge.py` 和 `src/kd_sensing/diagnostics/viewer_manifest_writer.py`。
- `src/kd_sensing/data/deepverse/label_builder.py` 保留 `DeepVerseLabelBuilder` 公开入口；label constants、scene metadata/config resolution、target derivation 和 cache writer 分别迁移到 `src/kd_sensing/data/deepverse/label_constants.py`、`src/kd_sensing/data/deepverse/label_scene.py`、`src/kd_sensing/data/deepverse/label_targets.py` 和 `src/kd_sensing/data/deepverse/label_writers.py`，split 与 sanity check 继续由 `src/kd_sensing/data/deepverse/split.py` 和 `src/kd_sensing/data/deepverse/sanity_check.py` 负责。
- `src/kd_sensing/data/mmw/preparation.py` 保留 Town10/Skybridge MMW preparation 公开 orchestration 和兼容导出；配置 schema、默认常量和 override loading 迁移到 `src/kd_sensing/data/mmw/preparation_config.py`，zip/input audit、extract marker、availability report 迁移到 `src/kd_sensing/data/mmw/preparation_audit.py`，sensor/channel indexing 与 path parsing 迁移到 `src/kd_sensing/data/mmw/preparation_index.py`，sequence row、group-safe split、guard band 和 leakage diagnostics 迁移到 `src/kd_sensing/data/mmw/preparation_splits.py`，channel payload、DFT/codebook beam power 和 power validation 迁移到 `src/kd_sensing/data/mmw/preparation_beam_power.py`，manifest/split/report 写出迁移到 `src/kd_sensing/data/mmw/preparation_writers.py`，relative geometry、pose/proxy features 和 azimuth bin helper 迁移到 `src/kd_sensing/data/mmw/preparation_geometry.py`。

新增内部代码不得从 `kd_sensing.engine.objective_metadata` 或 `kd_sensing.data.mmw.preparation` 回流导入窄 helper；应直接使用上面的窄模块。`kd_sensing.diagnostics.viewer_manifest`、`kd_sensing.data.deepverse.label_builder` 和 `kd_sensing.data.mmw.preparation` 可作为公开 orchestration/import 入口，但内部 helper 引用应分别指向 `viewer_manifest_*`、`label_*` 与 `preparation_*` 窄模块。

## 第二梯队热点

第二梯队热点先纳入 inventory 和架构 review 清单，不在本批次做大规模行为改写：

- HiST-Beam engine/model/evaluation 专用源码已退役并从当前支持面删除；旧 registry 名称和配置路径只保留 migration guard 错误信息。
- `src/kd_sensing/diagnostics/run_index.py`：后续优先抽出 process/resource collection、artifact summary、CSV/render writers；当前保持诊断输出 schema 兼容。
- `tools/visualization/gradio_multimodal_viewer.py`：后续优先抽出 render cache、filter/status helper 和 viewer state glue；当前作为 viewer_entrypoint 保持 import smoke 和 CLI 行为。
- `src/kd_sensing/data/transform_ops/csi.py`：后续优先抽出 CSI parsing、hardening feature transforms 和 temporal window helpers；当前避免同时改动数据契约。
- `src/kd_sensing/engine/batch.py`：后续优先抽出 modality target preparation、label adapters 和 history anchor input helper；当前保持训练 batch contract。
- `src/kd_sensing/engine/evaluation_pass.py`：后续优先抽出 metrics aggregation、objective-specific outputs 和 prediction metadata helper；当前保持 evaluation result schema。

## 配置 YAML

当前 `configs/fusion/` 根目录有 12 个实体 YAML。`configs/csi/hardening_matrix/` 有 13 个主矩阵 YAML，`configs/csi/hardening_matrix/debug/` 有 5 个 debug YAML；`configs/fusion/csi_hardening_matrix/` 有 4 个 GPS+CSI 验证矩阵 YAML。

已退役的 CRAF、MARF、G2D、Multimodal-NF 和 KD 实体 YAML、overlay recipe 与 virtual alias 不再作为支持入口存在。删除实体文件后，配置加载器只为当前 strong/lightweight canonical、snapshot、objective-aware 和保留 overlay 生成 virtual config，不接管退役路径；旧 `logits_kd` / `rkd` 路径只作为 migration guard 的拒绝命中保留。

## 脚本入口 Allowlist

保留入口按 lifecycle 分类如下；新增 `scripts/`、`tools/analysis/` 或 `tools/visualization/` 下的 Python/shell 文件必须同步更新本 inventory 和 `tests/test_architecture_boundaries.py`。

- package_cli: `kd_sensing.cli.inspect_deepsense6g_residual_inputs`、`kd_sensing.cli.prepare_deepsense6g_residual_manifest`、`kd_sensing.cli.run_deepsense6g_residual_fusion`、`kd_sensing.cli.plot_deepsense6g_residual_fusion`、`kd_sensing.cli.compare_deepsense6g_residual_with_gps_v2`。这些是 DeepSense6G GPS-prior anchored residual correction 的包内入口，对应 console scripts 为 `kd-sensing-inspect-deepsense6g-residual-inputs`、`kd-sensing-prepare-deepsense6g-residual-manifest`、`kd-sensing-run-deepsense6g-residual-fusion`、`kd-sensing-plot-deepsense6g-residual-fusion` 和 `kd-sensing-compare-deepsense6g-residual-with-gps-v2`；它们不新增顶层 `src.*` 模块，不作为从零多模态 beam prediction 入口。
- thin_cli_alias: `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`、`scripts/train_baseline.py`、`scripts/eval_baseline.py`、`scripts/train_beambench_image_ae_gps.py`、`scripts/run_beambench_image_ae_gps_tableiii.py`。前三者只委托包内主训练/评估/预处理 CLI；README 推荐 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。`scripts/train_baseline.py` 和 `scripts/eval_baseline.py` 是 BeamBench baseline 审计/mock smoke 薄入口，`scripts/train_beambench_image_ae_gps.py` 是 Arnold22 BeamBench Table III `Camera=AE, GPS=Direct, Fusion=Yes` 本地训练薄入口，`scripts/run_beambench_image_ae_gps_tableiii.py` 是四场景 Table III 本地复现实验薄入口；主要实现位于 `src/kd_sensing/baselines/beambench/` 和 `src/kd_sensing/cli/`，默认输出限定在 ignored 的 `outputs/beambench_baseline/`、`outputs/beambench_image_ae_gps_direct/` 或 `outputs/beambench_image_ae_gps_direct_tableiii/`。
- research_diagnostic: `scripts/analyze_csi_hardening_sweep.py`、`scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py`、`scripts/analysis/visualize_deepsense_beambench_correspondence.py`、`scripts/debug_eval_consistency.py`、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`、`scripts/mmw/visualize_gps_angle_beam_correspondence.py`、`scripts/mmw/visualize_gps_prediction_trajectory.py`、`scripts/mmw/visualize_prediction_error_label_distribution.py`。旧模态子集/扰动研究脚本不再作为长期入口；通用 subset/mask 验证保留在 `kd-sensing-evaluate` 使用的共享 evaluation pass 与配置化 `evaluation.modality_subsets` 中。
- dataset_preparation: `scripts/inspect_dataset.py`、`scripts/check_dataset.py`、`scripts/mmw/prepare_town10_skybridge.py`、`scripts/mmw/build_sequence_splits_from_manifest.py`、`scripts/mmw/visualize_town_label_distribution.py`。
- viewer_entrypoint: `tools/visualization/gradio_multimodal_viewer.py`。
- viewer_support: `tools/visualization/viewer_utils.py`、`tools/visualization/viewer_constants.py`、`tools/visualization/viewer_manifest_io.py`、`tools/visualization/viewer_figures.py`、`tools/visualization/viewer_prediction_tables.py`。
- shell_orchestration: `scripts/run_csi_hardening_matrix.sh`、`scripts/run_deepsense_gps_circular_soft_label.sh`、`scripts/run_mmw_gps_circular_soft_label_ablation.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`。

MMW 入口生命周期说明：

- `scripts/check_dataset.py` 属于 dataset_preparation。职责是只读检查 BeamBench/DeepSense6G CSV 字段、传感器路径引用、beam label 范围以及 scene/sample/sequence/timestamp 标识解析；输出可写入显式 JSON 报告，不移动、不删除、不生成真实数据。
- `scripts/train_baseline.py` 和 `scripts/eval_baseline.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.baselines.beambench` 中的 BeamBench 复现实现：前者打通 mock train/eval/checkpoint smoke，后者生成官方 `challenge.py` 评估计划或执行 mock checkpoint 评估。真实官方评估只有在官方数据、权重、源码和环境齐备且显式传入 `--execute` 时才运行。
- `scripts/train_beambench_image_ae_gps.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.baselines.beambench.image_ae_gps` 中的论文 row 专用实现：从本地 DeepSense6G scene31-34 sequence CSV 读取 camera/GPS/future beam，先训练或加载 Camera AE，再冻结 AE encoder 训练 GPS Direct concat fusion classifier，输出 checkpoint、history、predictions 和 BeamBench DBA/top-k metrics；输出限定在 `outputs/` 下，不得提交新 checkpoint、日志或 predictions。
- `scripts/run_beambench_image_ae_gps_tableiii.py` 属于 thin_cli_alias。职责是委托 `kd_sensing.cli.run_beambench_image_ae_gps_tableiii`，顺序运行 scene31-34 的 Camera AE + GPS Direct 本地复现实验并输出 Table III 风格 CSV/Markdown/JSON 汇总；输出限定在 `outputs/` 下，不得提交新 checkpoint、feature cache、predictions 或 summary runtime artifact。
- `scripts/analysis/visualize_deepsense_beambench_correspondence.py` 属于 research_diagnostic。职责是读取本地 DeepSense6G scene31-34 原始 scenario CSV、GPS 和 beam labels，输出 BeamBench Fig.2 风格的 calibrated GPS angle 与 centered beam index 空间对应图；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/analysis/deepsense_gps_v2_support_sweep_artifacts.py` 属于 research_diagnostic。职责是汇总 DeepSense6G GPS v2 support sweep 本地 artifact，输出只限 `outputs/analysis/` 等显式诊断路径，不得提交生成统计或图表产物。
- `scripts/mmw/build_sequence_splits_from_manifest.py` 属于 dataset_preparation。职责是在已有 `Prepared/<scene>/manifests/frame_manifest.csv` 基础上生成指定 `seq_len`/`pred_len` 的 sequence split CSV 和 `split_metadata.json`，服务于已完成 manifest 准备但需要补建 split 的本地数据准备流程。推荐长期入口仍是包内 MMW 数据准备能力或 `scripts/mmw/prepare_town10_skybridge.py`；该脚本是短期可审计的补充入口。输出仅允许写入 dataset 或显式本地数据根下的 `Prepared/<scene>/splits/<split_tag>/`，不得写入源码目录。删除/收敛条件是包内公开 split materialization utility 或 preprocessor CLI 覆盖同等参数、metadata 和错误提示后，将该脚本降级为 thin alias 或移除。
- `scripts/mmw/visualize_town_label_distribution.py` 属于 dataset_preparation。职责是读取本地 MMW Town split/manifest 数据并输出标签分布诊断图或摘要，辅助确认场景标签偏移；输出限定为显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_gps_angle_beam_correspondence.py` 属于 research_diagnostic。职责是读取本地 MMW Town split CSV 和 GPS anchor calibration summary，输出 BeamBench 风格的 GPS calibrated angle 与 mapping-centered beam index 空间对应图；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_gps_prediction_trajectory.py` 属于 research_diagnostic。职责是读取本地 MMW Town split CSV 与 GPS anchor `predictions.csv`，输出真实 beam、GPS 预测 beam、DBA 在实际空间轨迹和样本序列上的对照图，辅助定位 DBA=0 的空间/序列偏移来源；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/mmw/visualize_prediction_error_label_distribution.py` 属于 research_diagnostic。职责是读取本地预测 artifact 中的 `predictions.csv` 和 `summary.json`，输出预测错误样本的真实 beam label 分布图、源/目标场景标注和摘要；输出限定为 `outputs/analysis/` 等显式本地诊断路径，不得提交生成图片或统计产物。
- `scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 和 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 属于 shell_orchestration。职责是运行 sunny MMW 15 组 modal quick validation profile，分别固定对应 `seq_len`/`num_pred` 和 metric horizon 组合，并可选调用 split、radar map 和 cache 预热准备。默认输出根为 `outputs/mmw_sunny_modal15/<horizon_tag>/`；输出边界限定为 `outputs/`、`logs/`、dataset 准备产物和 cache/checkpoint 等本地运行产物，不得提交新生成结果。
- `scripts/run_mmw_gps_circular_soft_label_ablation.sh` 属于 shell_orchestration。职责是运行 sunny MMW GPS neural baseline 的 hard CE 与 circular Gaussian soft-label CE 对照实验，固定 MMW split、GPS-only 输入和 DBA 早停指标，用于诊断 beam codebook 边界/跳变对 GPS 监督的影响。输出边界限定为 `outputs/analysis/mmw_town_label_distribution/gps_circular_soft_label_ablation/`、`logs/mmw_gps_circular_soft_label_ablation/`、checkpoint 和本地训练缓存，不得提交新生成结果。
- `scripts/run_deepsense_gps_circular_soft_label.sh` 属于 shell_orchestration。职责是运行 DeepSense6G scene31-34 的 GPS-only circular Gaussian soft-label baseline，固定 DeepSense sequence CSV、GPS-only 输入和 DBA 早停指标，用于和 MMW Town GPS 监督诊断对照。输出边界限定为 `outputs/training/deepsense6g_gps_circular_soft_label/`、`logs/deepsense6g_gps_circular_soft_label/`、checkpoint 和本地训练缓存，不得提交新生成结果。
已退役的 image-only legal crossroad probe、P3/V8 批处理和等待式 shell wrapper 已从 allowlist 删除；历史本地输出只通过 runtime cleanup manifest 作为候选审计，不再作为当前入口维护。

`tools/visualization/export_viewer_manifest.py` 不得回流；`kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 已覆盖同一 manifest 导出 workflow。

## 本地产物

本地运行产物清理采用两阶段工作流：先运行 `kd-sensing-clean-runtime-artifacts` 生成 JSON manifest，再人工检查候选路径、规则、大小、mtime、风险等级和保护原因。真正删除必须复用 manifest 并显式传入 `--delete --manifest <path> --confirm-delete`；删除阶段会再次检查路径仍在扫描根内、未被 git 跟踪、未落入受保护根且状态没有相对 manifest 漂移。

`outputs/mmw_sunny_modal15/<horizon_tag>/` 是 MMW modal15 shell orchestration 的语义化默认输出命名约定。历史 `outputs/other/` 不自动迁移、不自动删除；它只作为清理 manifest 中的 `output.ambiguous_other` 人工确认候选出现，并保留 run index 的状态、checkpoint 数量和大小摘要。

本 change 不移动、删除、压缩或重写真实数据与本地实验产物。架构边界测试只检查已跟踪路径，继续拒绝：

- `__pycache__`、`.pyc`、`.pytest_cache`
- `outputs/`、`logs/`
- 除 `dataset/.gitkeep` 之外的 `dataset/` 内容
- 非 `All_models/` 历史资料范围内的 `.pth`、`.pt`、`.ckpt`

`dataset/.gitkeep` 是允许的源码占位文件。
