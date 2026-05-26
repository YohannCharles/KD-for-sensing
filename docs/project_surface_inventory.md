# 项目表面积 Inventory

本 inventory 记录 `refine-source-architecture-and-entry-surface` 的可审计基线。统计口径只覆盖源码、配置、文档和 OpenSpec artifact；`dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包和其它本地运行产物不属于本 change 的处理范围。

## 源码热点模块

本批次拆分的热点 facade 与职责模块如下：

- `tools/visualization/viewer_utils.py` 保留兼容导出；manifest 读取、路径解析和 scene/split/show mode 过滤迁移到 `tools/visualization/viewer_manifest_io.py`，图表构造迁移到 `tools/visualization/viewer_figures.py`，prediction summary 和 legacy prediction adapter 迁移到 `tools/visualization/viewer_prediction_tables.py`，常量迁移到 `tools/visualization/viewer_constants.py`。
- `src/kd_sensing/preprocessing/raymobtime_s008.py` 保留 preprocessor registry；paths/audit、index、beam labels、ray features 和 cache writer 分别迁移到 `raymobtime_s008_paths.py`、`raymobtime_s008_index.py`、`raymobtime_s008_beam_labels.py`、`raymobtime_s008_ray_features.py`、`raymobtime_s008_cache.py`，共享常量和窄 helper 在 `raymobtime_s008_common.py`。
- `src/kd_sensing/models/csi.py` 保留公开 import 路径；pilot estimation、CSI hardening、view tokenizer/fusion、debug helpers 和 encoder registry glue 分别迁移到 `csi_estimation.py`、`csi_hardening.py`、`csi_views.py`、`csi_debug.py`、`csi_encoder.py`。
- `src/kd_sensing/engine/objective_metadata.py` 保留公开兼容 facade；objective 名称、默认 metric、metric alias 和 mode 表迁移到 `src/kd_sensing/engine/objectives/registry.py`，history fields 与 TensorBoard scalar schema 迁移到 `src/kd_sensing/engine/objectives/history.py`，runtime metadata/validation helper 在 `src/kd_sensing/engine/objectives/metadata.py`。
- `src/kd_sensing/preprocessing/multimodal_nf_common.py` 保留公开兼容 facade；constants、path resolution、audit、HDF5 inspection、codebook metadata、split assignment 和 index writer/loader 分别迁移到 `src/kd_sensing/preprocessing/multimodal_nf_constants.py`、`src/kd_sensing/preprocessing/multimodal_nf_paths.py`、`src/kd_sensing/preprocessing/multimodal_nf_audit.py`、`src/kd_sensing/preprocessing/multimodal_nf_hdf5.py`、`src/kd_sensing/preprocessing/multimodal_nf_codebook.py`、`src/kd_sensing/preprocessing/multimodal_nf_splits.py` 和 `src/kd_sensing/preprocessing/multimodal_nf_index.py`。
- `src/kd_sensing/diagnostics/viewer_manifest.py` 保留 manifest 导出公开 orchestration；sample id/JSON schema、cache metadata、row path resolution、prediction/quality/gate merge 和 asset writer 分别迁移到 `src/kd_sensing/diagnostics/viewer_manifest_schema.py`、`src/kd_sensing/diagnostics/viewer_manifest_cache.py`、`src/kd_sensing/diagnostics/viewer_manifest_paths.py`、`src/kd_sensing/diagnostics/viewer_manifest_merge.py` 和 `src/kd_sensing/diagnostics/viewer_manifest_writer.py`。
- `src/kd_sensing/data/deepverse/label_builder.py` 保留 `DeepVerseLabelBuilder` 公开入口；label constants、scene metadata/config resolution、target derivation 和 cache writer 分别迁移到 `src/kd_sensing/data/deepverse/label_constants.py`、`src/kd_sensing/data/deepverse/label_scene.py`、`src/kd_sensing/data/deepverse/label_targets.py` 和 `src/kd_sensing/data/deepverse/label_writers.py`，split 与 sanity check 继续由 `src/kd_sensing/data/deepverse/split.py` 和 `src/kd_sensing/data/deepverse/sanity_check.py` 负责。

新增内部代码不得从 `kd_sensing.engine.objective_metadata` 或 `kd_sensing.preprocessing.multimodal_nf_common` 回流导入；应直接使用上面的窄模块。`kd_sensing.diagnostics.viewer_manifest` 和 `kd_sensing.data.deepverse.label_builder` 可作为公开 orchestration/import 入口，但内部 helper 引用应指向 `viewer_manifest_*` 与 `label_*` 窄模块。

## 配置 YAML

当前 `configs/fusion/` 根目录有 19 个实体 YAML。`configs/csi/hardening_matrix/` 有 13 个主矩阵 YAML，`configs/csi/hardening_matrix/debug/` 有 5 个 debug YAML；`configs/fusion/csi_hardening_matrix/` 的 GPS+CSI 验证矩阵暂不删除。

已 recipe 化并防止回流的 virtual alias：

- G2D：`configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`、`configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`、`configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`。
- CRAF：`configs/fusion/craf_all_modalities_no_kd.yaml`、`configs/fusion/craf_all_modalities_no_counterfactual.yaml`、`configs/fusion/craf_all_modalities_fixed_prior_sanity.yaml`。
- MARF：`configs/fusion/marf.yaml`、`configs/fusion/marf_subset_training.yaml`、`configs/fusion/marf_no_residual_ablation.yaml`、`configs/fusion/marf_no_prior_bias_ablation.yaml`、`configs/fusion/marf_no_subset_training_ablation.yaml`。

这些路径由 `src/kd_sensing/config/canonical_recipes/advanced.py` 的 advanced overlay alias 生成；删除实体文件后仍可直接加载同名 virtual config，并保持 experiment、task、dataset、modalities、model、loss/distillation、training、run_name 和 checkpoint 来源等关键语义一致。

继续保留的高级实体 YAML 分类：

- 可由 recipe 生成但存在显式差异：`craf_all_modalities_stabilized_no_kd.yaml`、`craf_image_radar_no_kd.yaml`。差异涉及稳定化训练、模态集合或 reliability/training 字段。
- 需要作为人工样例继续保留：teacher-prior CRAF/stage 配置、token transformer 样例、CSI/GPS/mmWave 组合和 legacy named examples。后续删除前必须先补等价检查或记录允许差异。

## 脚本入口 Allowlist

保留入口按 lifecycle 分类如下；新增 `scripts/`、`tools/analysis/` 或 `tools/visualization/` 下的 Python/shell 文件必须同步更新本 inventory 和 `tests/test_architecture_boundaries.py`。

- thin_cli_alias: `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`。这些只委托包内 CLI；README 推荐 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。
- research_diagnostic: `scripts/analyze_csi_hardening_sweep.py`、`scripts/build_teacher_registry.py`、`scripts/debug_eval_consistency.py`、`scripts/eval_modality_perturbation.py`、`scripts/eval_modality_subsets.py`、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`。
- dataset_preparation: `scripts/deepverse/download_dt31_assets.py`、`scripts/deepverse/generate_dt31_cache.py`、`scripts/mmw/prepare_town10_skybridge.py`。
- viewer_entrypoint: `tools/visualization/gradio_multimodal_viewer.py`。
- viewer_support: `tools/visualization/viewer_utils.py`、`tools/visualization/viewer_constants.py`、`tools/visualization/viewer_manifest_io.py`、`tools/visualization/viewer_figures.py`、`tools/visualization/viewer_prediction_tables.py`。
- shell_orchestration: `scripts/run_csi_hardening_matrix.sh`。

`tools/visualization/export_viewer_manifest.py` 不得回流；`kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 已覆盖同一 manifest 导出 workflow。

## 本地产物

本 change 不移动、删除、压缩或重写真实数据与本地实验产物。架构边界测试只检查已跟踪路径，继续拒绝：

- `__pycache__`、`.pyc`、`.pytest_cache`
- `outputs/`、`logs/`
- 除 `dataset/.gitkeep` 之外的 `dataset/` 内容
- 非 `All_models/` 历史资料范围内的 `.pth`、`.pt`、`.ckpt`

`dataset/.gitkeep` 是允许的源码占位文件。
