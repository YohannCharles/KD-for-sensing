## Why

项目已经决定彻底放弃“模态失衡”研究线，现有 Conditional Utility、弱模态互补样本、Phase 1.5、Raymobtime s008 失衡诊断和 G2D 失衡汇总代码继续占据 CLI、viewer、测试和 OpenSpec 表面积，容易让后续 agent 与开发者误以为这些研究流程仍是当前主线。

本变更用于收敛源码与规格：移除失衡研究专属入口、分析模块、配置、测试和文档要求，同时保留基础多模态训练、Raymobtime s008 数据/模型能力、通用 subset/mask 能力，以及仍被 CSI hardening 等流程复用的 G2D-style 训练能力。

## What Changes

- **BREAKING**：退役 Conditional Utility Audit 研究流程，不再提供 `tools/analysis/run_conditional_utility_audit.py`、`tools/analysis/analyze_conditional_utility.py`、`configs/analysis/marf_conditional_utility_audit.yaml` 及其输出 schema 契约。
- **BREAKING**：退役弱模态互补样本分析和 Gradio Complementarity Explorer，不再提供 `scripts/analysis/build_complementarity_cases.py`、`tools/visualization/complementarity_explorer.py`、viewer 的 `--complementarity-dir` 参数和对应 Tab。
- **BREAKING**：退役 Phase 1.5 Utility Validation，不再提供 Phase 1.5 manifest、bootstrap CI、checkpoint matrix、fixed-subset baseline matrix 或决策报告。
- **BREAKING**：退役 Raymobtime s008 模态失衡诊断分析入口和诊断矩阵，不再要求 `kd-sensing-raymobtime-analysis`、`src/kd_sensing/diagnostics/raymobtime_analysis.py` 或 `configs/raymobtime/s008_modality_imbalance_diagnosis.yaml`。
- 移除 G2D 多模态失衡结果汇总脚本要求，但保留 G2D distiller、teacher ensemble、SMP、epoch diagnostics、advanced overlay 和 GPS+CSI G2D-style 配置能力。
- 保留 Raymobtime s008 预处理、dataset、current snapshot 模型、单任务/多任务训练和评估指标；仅删除“失衡诊断/判定/外部验证门槛”这层研究解释工具。
- 保留 `force_modality_mask`、通用 `evaluation.modality_subsets`、CRAF/MARF subset/counterfactual 训练所需 mask 能力；仅移除 Conditional Utility 专属 subset registry 和命名契约。
- 同步收敛 README、docs、OpenSpec、架构边界测试、脚本 allowlist 和健康检查，避免已退役入口继续作为必须存在的项目契约。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `conditional-utility-audit`: 退役 Conditional Utility Audit 全部研究需求与产物契约。
- `weak-modality-complementarity-analysis`: 退役互补 case mining、summary、report、Explorer 和相关自动化测试需求。
- `phase-1-5-utility-validation`: 退役 Phase 1.5 运行清单、bootstrap、checkpoint matrix、baseline matrix 和决策报告需求。
- `raymobtime-modality-imbalance-diagnosis`: 退役 Raymobtime s008 模态失衡诊断矩阵、内部机制证据、判定标准和 s009 外部验证门槛。
- `raymobtime-s008-selection`: 删除 Raymobtime s008 规格中“模态失衡分析”要求，保留数据审计、cache、dataset、snapshot 模型、训练和评估能力。
- `experiment-workflow`: 删除 Raymobtime 模态失衡分析 workflow、Phase 1.5/互补分析健康检查和相关文档要求，保留普通训练评估、G2D training workflow 与 future horizon metrics。
- `g2d-multimodal-distillation`: 删除 G2D 多模态失衡结果汇总脚本要求，保留 G2D 训练、teacher ensemble、SMP 和 diagnostics artifact。
- `project-architecture`: 更新入口和健康检查契约，移除 `kd-sensing-raymobtime-analysis`、互补性 facade 分层要求和 Phase 1.5/互补分析快速检查要求。

## Impact

- 影响源码：`src/kd_sensing/diagnostics/conditional_utility.py`、`src/kd_sensing/diagnostics/complementarity*.py`、`src/kd_sensing/diagnostics/phase_1_5_utility_validation.py`、`src/kd_sensing/diagnostics/raymobtime_analysis.py`、`src/kd_sensing/cli/raymobtime_analysis.py`、`tools/analysis/*conditional*`、`tools/analysis/run_phase_1_5_utility_validation.py`、`tools/analysis/collect_multimodal_imbalance_results.py`、`tools/visualization/complementarity_explorer.py`、`scripts/analysis/build_complementarity_cases.py`。
- 影响 viewer：删除 Complementarity Explorer import、参数、状态、Tab、回调和测试；保留 manifest viewer、raw/processed modalities、diagnostics、future distribution 和 manifest 导出流程。
- 影响配置：删除 `configs/analysis/marf_conditional_utility_audit.yaml`、`configs/analysis/phase_1_5_utility_validation.yaml`、`configs/raymobtime/s008_modality_imbalance_diagnosis.yaml`；不删除 G2D overlay、CSI hardening matrix 或 Raymobtime s008 train/preprocess 配置。
- 影响测试与文档：删除或改写对应测试，更新 `tests/test_architecture_boundaries.py`、README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md`、`docs/research_notes.md`、`tools/visualization/README.md` 和相关 OpenSpec。
- 不迁移、不删除本地 `outputs/`、`logs/`、`dataset/`、cache、checkpoint 或历史实验产物；只移除源码、配置、测试和文档中的已退役研究表面积。
