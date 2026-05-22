# 项目表面积 Inventory

本 inventory 对 `reduce-redundant-project-surface` 变更中的删除、保留和检查项给出可审计基线。统计口径只覆盖源码树，不把本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `__pycache__` 作为可提交表面积。

## 配置 YAML

实施前 `configs/fusion/` 有 30 个实体 YAML。第一批删除候选选择 3 个 G2D 五模态实体配置：

- `configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- `configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- `configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`

这些路径由 canonical advanced overlay recipe 生成，删除实体文件后仍可直接加载同名 virtual config，并保持 experiment、task、dataset、modalities、model、loss/distillation、training、run_name 和 teacher checkpoint 来源一致。

本批次保留的高级 fusion 实体配置：

- CRAF：`craf_all_modalities_no_kd.yaml`、`craf_all_modalities_no_counterfactual.yaml`、`craf_all_modalities_fixed_prior_sanity.yaml`、`craf_all_modalities_stabilized_no_kd.yaml`、`craf_image_radar_no_kd.yaml`。保留原因是现有 overlay 与实体配置在模型 reliability、loss、training schedule 或 dataset 字段上还存在显式差异。
- MARF：`marf.yaml`、`marf_subset_training.yaml`、`marf_no_residual_ablation.yaml`、`marf_no_prior_bias_ablation.yaml`、`marf_no_subset_training_ablation.yaml`。保留原因是实体配置仍记录 ablation 专用字段，例如 `random_keep_prob` 和 subset evaluation 差异。
- Teacher-prior CRAF / stage 配置、token transformer、CSI/GPS/mmWave 组合和 legacy named examples 继续作为人工维护入口保留。

`configs/csi/hardening_matrix/` 有 13 个主矩阵 YAML，`configs/csi/hardening_matrix/debug/` 有 5 个 debug YAML；本变更只 inventory，不删除。`configs/fusion/csi_hardening_matrix/` 的 GPS+CSI 验证矩阵同样暂不删除。

## 脚本入口

实施前 Python 脚本入口统计：

- `scripts/`：14 个 Python 文件。
- `tools/analysis/`：4 个 Python 文件。
- `tools/visualization/`：4 个 Python 文件。

本变更删除 `tools/visualization/export_viewer_manifest.py`，因为 `kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 已覆盖同一 manifest 导出工作流。

保留入口按生命周期分类：

- 薄 CLI alias：`scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py`。这些只委托包内 CLI，保留为现有本地命令兼容入口；README 推荐 console script。
- 研究/诊断脚本：CSI hardening sweep、teacher registry、eval perturbation/subsets、training I/O profile、parallel training recommendation、conditional utility 和 complementarity 分析脚本。
- 数据准备脚本：DeepVerse DT31、MMW Town10 skybridge 相关准备脚本。
- Viewer 支持：`tools/visualization/gradio_multimodal_viewer.py`、`complementarity_explorer.py`、`viewer_utils.py`。
- Shell orchestration：`scripts/run_csi_hardening_matrix.sh`。

新增或保留入口必须同步更新 `tests/test_architecture_boundaries.py` 中的 allowlist，并通过 OpenSpec change 说明原因。

## README / OpenSpec

实施前 README 为 901 行，承载了 G2D、CRAF/MARF、CSI hardening、viewer 和 Raymobtime 的详细实验说明。本变更将 README 收缩为入口地图，并把实验矩阵集中到 `docs/experiment_matrix.md`，viewer 细节继续放在 `tools/visualization/README.md`。

实施前 `openspec/specs/*/spec.md` 中存在多个 `TBD - created by archiving` purpose。本变更补齐当前 specs 的真实 purpose，并增加架构检查拒绝 TBD purpose 回流。

## 本地产物

本地工作区可能存在未跟踪的 `__pycache__`、`.pytest_cache`、`outputs/`、`logs/` 和数据 cache。这些不是源码表面积，不应提交。架构边界测试只检查已跟踪路径，拒绝：

- `__pycache__`、`.pyc`、`.pytest_cache`
- `outputs/`、`logs/`
- 除 `dataset/.gitkeep` 之外的 `dataset/` 内容
- 非 `All_models/` 历史资料范围内的 `.pth`、`.pt`、`.ckpt`

`dataset/.gitkeep` 是允许的源码占位文件。
