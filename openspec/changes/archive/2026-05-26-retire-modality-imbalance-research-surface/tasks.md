## 1. 规格与删除边界确认

- [x] 1.1 对照本 change 的 specs，确认退役范围仅覆盖 Conditional Utility、Complementarity、Phase 1.5、Raymobtime imbalance analysis 和 G2D imbalance result collector
- [x] 1.2 确认可保留能力清单：Raymobtime s008 preprocess/dataset/model/training/evaluation、G2D distiller/SMP/training extension、CSI hardening G2D-style 配置、`force_modality_mask`、通用 `evaluation.modality_subsets`
- [x] 1.3 运行 `openspec validate retire-modality-imbalance-research-surface --strict`，先保证变更规格本身合法

## 2. 删除退役研究源码与入口

- [x] 2.1 删除 Conditional Utility 代码、脚本和配置：`src/kd_sensing/diagnostics/conditional_utility.py`、`tools/analysis/run_conditional_utility_audit.py`、`tools/analysis/analyze_conditional_utility.py`、`configs/analysis/marf_conditional_utility_audit.yaml`
- [x] 2.2 删除 Complementarity 后端、脚本和 viewer support：`src/kd_sensing/diagnostics/complementarity*.py`、`scripts/analysis/build_complementarity_cases.py`、`tools/visualization/complementarity_explorer.py`
- [x] 2.3 删除 Phase 1.5 代码、脚本和配置：`src/kd_sensing/diagnostics/phase_1_5_utility_validation.py`、`tools/analysis/run_phase_1_5_utility_validation.py`、`configs/analysis/phase_1_5_utility_validation.yaml`
- [x] 2.4 删除 Raymobtime 模态失衡分析代码、CLI 和配置：`src/kd_sensing/diagnostics/raymobtime_analysis.py`、`src/kd_sensing/cli/raymobtime_analysis.py`、`configs/raymobtime/s008_modality_imbalance_diagnosis.yaml`
- [x] 2.5 删除 G2D 多模态失衡结果汇总脚本 `tools/analysis/collect_multimodal_imbalance_results.py`，但保留 G2D 训练、SMP、diagnostics 和 CSI hardening 配置
- [x] 2.6 更新 `src/kd_sensing/diagnostics/__init__.py`、`src/kd_sensing/evaluation/subset_specs.py` 及调用方，移除 Conditional Utility/Complementarity 专属 lazy export 和 subset registry，同时保持 validator 的通用 subset 名称可用
- [x] 2.7 更新 `pyproject.toml`，移除 `kd-sensing-raymobtime-analysis` console script，保留 train/evaluate/preprocess/runs/manifest/visualize 入口

## 3. 收敛 Gradio Viewer 与通用评估路径

- [x] 3.1 从 `tools/visualization/gradio_multimodal_viewer.py` 删除 `--complementarity-dir` 参数、complementarity import、数据加载、状态、Tab、筛选回调、导出回调和样本详情联动
- [x] 3.2 确认 manifest viewer 的 Overview、Raw Modalities、Processed Modalities、Diagnostics、future distribution 和启动参数继续可用
- [x] 3.3 确认 `src/kd_sensing/engine/validator.py` 的 `evaluation.modality_subsets` 仍支持 `all`、`strong_only`、`weak_only`、单模态名、组合名和 `drop_*`，但不依赖 Conditional Utility registry
- [x] 3.4 保留 `scripts/eval_modality_subsets.py`、`scripts/eval_modality_perturbation.py` 和 `scripts/debug_eval_consistency.py`，只在必要时更新它们的 import 或文案

## 4. 测试与架构边界更新

- [x] 4.1 删除退役测试：`tests/test_complementarity_analysis.py`、`tests/test_gradio_complementarity_explorer.py`、`tests/test_conditional_utility_metrics.py`、`tests/test_conditional_utility_oracle.py`、`tests/test_phase_1_5_utility_validation.py`
- [x] 4.2 从 `tests/test_raymobtime_s008_selection.py` 移除只覆盖 `analyze_raymobtime_modality_imbalance` 的测试，保留 Raymobtime s008 dataset/model/objective/training/evaluation 测试
- [x] 4.3 从 `tests/test_subset_specs.py` 移除 Conditional Utility subset registry 测试；如 validator 仍需覆盖通用 subset 解析，则迁移为 validator 或 evaluation subset 测试
- [x] 4.4 更新 `tests/test_architecture_boundaries.py` 的 Python entrypoint allowlist、retired/generated path 检查、diagnostics facade 检查、console script help 期望和项目健康检查期望
- [x] 4.5 确认 G2D 定向测试仍覆盖训练本体：`tests/test_g2d_loss.py`、`tests/test_g2d_distiller.py`、`tests/test_g2d_smp.py`、`tests/test_g2d_diagnostics.py`

## 5. 文档与 OpenSpec 收敛

- [x] 5.1 更新 README，移除 Phase 1.5、Complementarity 和退役研究测试命令，保留当前推荐健康检查和主要 workflow
- [x] 5.2 更新 `docs/experiment_matrix.md`，删除 G2D 多模态失衡汇总、Raymobtime 模态失衡分析和弱模态研究路线描述，保留 G2D 训练与 CSI hardening G2D-style 内容
- [x] 5.3 更新 `docs/project_surface_inventory.md`，移除退役脚本、viewer support、diagnostics facade 和 console script 分类
- [x] 5.4 更新 `docs/research_notes.md` 和 `tools/visualization/README.md`，删除 Conditional Utility、弱模态互补样本、Complementarity Explorer 和 Phase 1.5 操作说明
- [x] 5.5 检查 `openspec/specs/` 中被本 change 修改的 capability，不再要求退役入口、脚本、viewer Tab 或研究报告存在

## 6. 验证

- [x] 6.1 运行 `openspec validate retire-modality-imbalance-research-surface --strict`
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py tests/test_modality_visual_diagnostics.py -q`
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py -q`
- [x] 6.5 运行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- [x] 6.6 如时间允许，运行最终回归 `conda run -n kd_mm_beam pytest -q`
