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
- `src/kd_sensing/engine/hist_beam_loso_execution.py` 保留 HiST-Beam LOSO 执行公开 facade 和顶层 `execute_loso_run_plan` 编排；preflight 迁移到 `src/kd_sensing/engine/hist_beam_loso_preflight.py`，stage executor 与 stage loop 迁移到 `src/kd_sensing/engine/hist_beam_loso_stages.py`，run identity/record/stage metadata 迁移到 `src/kd_sensing/engine/hist_beam_loso_records.py`，summary/progress/JSONL artifact 写出迁移到 `src/kd_sensing/engine/hist_beam_loso_artifacts.py`，scene/stage cfg、enabled modalities、prototype/reuse/cache key 和 throughput metadata 迁移到 `src/kd_sensing/engine/hist_beam_loso_config.py`，quick summary/conclusion eligibility helper 迁移到 `src/kd_sensing/engine/hist_beam_loso_summary.py`，matrix metadata summary 迁移到 `src/kd_sensing/engine/hist_beam_loso_matrix.py`。
- `src/kd_sensing/data/mmw/preparation.py` 保留 Town10/Skybridge MMW preparation 公开 orchestration 和兼容导出；配置 schema、默认常量和 override loading 迁移到 `src/kd_sensing/data/mmw/preparation_config.py`，zip/input audit、extract marker、availability report 迁移到 `src/kd_sensing/data/mmw/preparation_audit.py`，sensor/channel indexing 与 path parsing 迁移到 `src/kd_sensing/data/mmw/preparation_index.py`，sequence row、group-safe split、guard band 和 leakage diagnostics 迁移到 `src/kd_sensing/data/mmw/preparation_splits.py`，channel payload、DFT/codebook beam power 和 power validation 迁移到 `src/kd_sensing/data/mmw/preparation_beam_power.py`，manifest/split/report 写出迁移到 `src/kd_sensing/data/mmw/preparation_writers.py`，relative geometry、pose/proxy features 和 azimuth bin helper 迁移到 `src/kd_sensing/data/mmw/preparation_geometry.py`。

新增内部代码不得从 `kd_sensing.engine.objective_metadata`、`kd_sensing.engine.hist_beam_loso_execution` 或 `kd_sensing.data.mmw.preparation` 回流导入窄 helper；应直接使用上面的窄模块。`kd_sensing.diagnostics.viewer_manifest`、`kd_sensing.data.deepverse.label_builder`、`kd_sensing.engine.hist_beam_loso_execution` 和 `kd_sensing.data.mmw.preparation` 可作为公开 orchestration/import 入口，但内部 helper 引用应分别指向 `viewer_manifest_*`、`label_*`、`hist_beam_loso_*` 与 `preparation_*` 窄模块。

## 第二梯队热点

第二梯队热点先纳入 inventory 和架构 review 清单，不在本批次做大规模行为改写：

- `src/kd_sensing/models/fusion/hist_beam.py`：后续优先抽出 config normalization、adapter/head builders 和 hierarchical helper；当前包含 V7 shared/private residual 等仍在变化的模型逻辑，本批次只保留现有公开 model registry 行为。
- `src/kd_sensing/diagnostics/run_index.py`：后续优先抽出 process/resource collection、artifact summary、CSV/render writers；当前保持诊断输出 schema 兼容。
- `tools/visualization/gradio_multimodal_viewer.py`：后续优先抽出 render cache、filter/status helper 和 viewer state glue；当前作为 viewer_entrypoint 保持 import smoke 和 CLI 行为。
- `src/kd_sensing/data/transform_ops/csi.py`：后续优先抽出 CSI parsing、hardening feature transforms 和 temporal window helpers；当前避免同时改动数据契约。
- `src/kd_sensing/engine/batch.py`：后续优先抽出 modality target preparation、label adapters 和 history anchor input helper；当前保持训练 batch contract。
- `src/kd_sensing/engine/evaluation_pass.py`：后续优先抽出 metrics aggregation、objective-specific outputs 和 prediction metadata helper；当前保持 evaluation result schema。

## 配置 YAML

当前 `configs/fusion/` 根目录有 11 个实体 YAML。`configs/csi/hardening_matrix/` 有 13 个主矩阵 YAML，`configs/csi/hardening_matrix/debug/` 有 5 个 debug YAML；`configs/fusion/csi_hardening_matrix/` 有 4 个 GPS+CSI 验证矩阵 YAML。

已退役的 CRAF、MARF、G2D 和 Multimodal-NF 实体 YAML、overlay recipe 和 virtual alias 不再作为支持入口存在。删除实体文件后，配置加载器只为当前 canonical、snapshot、objective-aware 和保留 overlay 生成 virtual config，不接管退役路径。

## 脚本入口 Allowlist

保留入口按 lifecycle 分类如下；新增 `scripts/`、`tools/analysis/` 或 `tools/visualization/` 下的 Python/shell 文件必须同步更新本 inventory 和 `tests/test_architecture_boundaries.py`。

- thin_cli_alias: `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`。这些只委托包内 CLI；README 推荐 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。
- research_diagnostic: `scripts/analyze_csi_hardening_sweep.py`、`scripts/debug_eval_consistency.py`、`scripts/eval_modality_perturbation.py`、`scripts/eval_modality_subsets.py`、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`。
- dataset_preparation: `scripts/inspect_dataset.py`、`scripts/mmw/prepare_town10_skybridge.py`、`scripts/mmw/build_sequence_splits_from_manifest.py`。
- viewer_entrypoint: `tools/visualization/gradio_multimodal_viewer.py`。
- viewer_support: `tools/visualization/viewer_utils.py`、`tools/visualization/viewer_constants.py`、`tools/visualization/viewer_manifest_io.py`、`tools/visualization/viewer_figures.py`、`tools/visualization/viewer_prediction_tables.py`。
- shell_orchestration: `scripts/run_csi_hardening_matrix.sh`、`scripts/run_mmw_sunny_modal15_l5p3_h123.sh`、`scripts/run_mmw_sunny_modal15_l5p6_h246.sh`、`scripts/run_p3_v8_fixed_source_skybridge_budget10_seed01_4gpu.sh`、`scripts/watch_modal15_then_run_p3.sh`。

MMW 入口生命周期说明：

- `scripts/mmw/build_sequence_splits_from_manifest.py` 属于 dataset_preparation。职责是在已有 `Prepared/<scene>/manifests/frame_manifest.csv` 基础上生成指定 `seq_len`/`pred_len` 的 sequence split CSV 和 `split_metadata.json`，服务于已完成 manifest 准备但需要补建 split 的本地数据准备流程。推荐长期入口仍是包内 MMW 数据准备能力或 `scripts/mmw/prepare_town10_skybridge.py`；该脚本是短期可审计的补充入口。输出仅允许写入 dataset 或显式本地数据根下的 `Prepared/<scene>/splits/<split_tag>/`，不得写入源码目录。删除/收敛条件是包内公开 split materialization utility 或 preprocessor CLI 覆盖同等参数、metadata 和错误提示后，将该脚本降级为 thin alias 或移除。
- `scripts/run_mmw_sunny_modal15_l5p3_h123.sh` 和 `scripts/run_mmw_sunny_modal15_l5p6_h246.sh` 属于 shell_orchestration。职责是运行 sunny MMW 15 组 modal quick validation profile，分别固定对应 `seq_len`/`num_pred` 和 metric horizon 组合，并可选调用 split、radar map 和 cache 预热准备。输出边界限定为 `outputs/`、`logs/`、dataset 准备产物和 cache/checkpoint 等本地运行产物，不得提交新生成结果。
- `scripts/run_p3_v8_fixed_source_skybridge_budget10_seed01_4gpu.sh` 和 `scripts/watch_modal15_then_run_p3.sh` 属于 shell_orchestration。职责是编排特定 MMW/HiST-Beam 本地实验批次和依赖等待，不作为包内 CLI 兼容承诺。删除/收敛条件是 HiST-Beam/MMW 包内 CLI 或矩阵配置能原生表达同等调度、metadata 与准备步骤时，将 shell wrapper 收敛为薄 alias 或移除。

`tools/visualization/export_viewer_manifest.py` 不得回流；`kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 已覆盖同一 manifest 导出 workflow。

## 本地产物

本 change 不移动、删除、压缩或重写真实数据与本地实验产物。架构边界测试只检查已跟踪路径，继续拒绝：

- `__pycache__`、`.pyc`、`.pytest_cache`
- `outputs/`、`logs/`
- 除 `dataset/.gitkeep` 之外的 `dataset/` 内容
- 非 `All_models/` 历史资料范围内的 `.pth`、`.pt`、`.ckpt`

`dataset/.gitkeep` 是允许的源码占位文件。
