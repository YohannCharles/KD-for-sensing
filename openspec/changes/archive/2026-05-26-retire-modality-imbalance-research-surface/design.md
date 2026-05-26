## Context

当前仓库中“模态失衡”研究线已经形成一组相互依赖的显式表面积：

- Scene32 Conditional Utility Audit 负责生成 subset prediction、delta、oracle、teacher complementarity、bucket 和图表。
- 弱模态互补样本分析从 audit 产物继续生成 case table、summary、report，并接入 Gradio viewer 的 Complementarity Explorer。
- Phase 1.5 Utility Validation 继续在 audit 产物之上做 bootstrap、checkpoint matrix、fixed-subset baseline matrix 和路线决策。
- Raymobtime s008 另有模态失衡诊断矩阵、分析 CLI、报告判定和 s009 外部验证门槛。
- G2D 既包含通用训练能力，也包含一个专门用于“多模态失衡结果汇总”的研究脚本。

用户已经明确放弃研究模态失衡问题，因此这些研究诊断工具会成为维护噪声。与此同时，项目仍需要保留普通多模态训练、Raymobtime s008 数据/模型/评估能力、CRAF/MARF 的 mask/subset 训练能力，以及被 CSI hardening 矩阵复用的 G2D-style 训练能力。

## Goals / Non-Goals

**Goals:**

- 移除失衡研究专属代码、CLI、脚本、配置、viewer Tab、测试、文档和 OpenSpec 契约。
- 保留可复用训练与评估能力：Raymobtime s008 current snapshot workflow、G2D distiller/SMP/diagnostics、CSI hardening G2D-style 配置、`force_modality_mask` 和通用 `evaluation.modality_subsets`。
- 让 README、docs、architecture boundary tests、entrypoint allowlist 和 OpenSpec 不再要求已退役研究入口。
- 不触碰本地实验产物、数据集、日志、checkpoint 或 cache。

**Non-Goals:**

- 不重写多模态 fusion、CRAF、MARF、G2D 或 Raymobtime 模型结构。
- 不删除 G2D 训练本体或 CSI hardening 的 GPS+CSI G2D-style 验证配置。
- 不删除 Raymobtime s008 预处理、dataset、cache、模型、训练和评估指标。
- 不迁移或清理 `outputs/`、`logs/`、`dataset/`、`All_models/` 中的本地产物或历史资料。

## Decisions

### 1. 以研究流程为边界删除，而不是按关键词删除

选择删除 Conditional Utility、Complementarity、Phase 1.5 和 Raymobtime imbalance diagnosis 这些完整研究流程。保留 `force_modality_mask`、通用 subset evaluation 和 G2D 训练，因为它们已被 CRAF/MARF/CSI 或普通调试路径复用。

替代方案是删除所有带 `weak/strong/imbalance/complementarity` 语义的代码。该方案会误伤通用模态子集评估、mask forward、G2D-style 训练和 Raymobtime 多任务评估，因此不采用。

### 2. G2D 只删除“失衡结果汇总脚本”，保留训练方法

`tools/analysis/collect_multimodal_imbalance_results.py` 是研究汇总入口，应退役。`src/kd_sensing/distillation/g2d.py`、`src/kd_sensing/distillation/g2d_smp.py`、`src/kd_sensing/engine/g2d_training.py` 和 `src/kd_sensing/diagnostics/g2d_diagnostics.py` 仍承担 `distillation.type: g2d` 训练、SMP 和 epoch diagnostics 合同，且 GPS+CSI hardening matrix 仍要求 G2D-style 配置。

替代方案是完全删除 G2D。该方案会破坏 `g2d-multimodal-distillation`、`experiment-workflow` 和 `csi-hardening-experiment-matrix` 仍保留的训练合同，因此不采用。

### 3. Raymobtime s008 删除分析 CLI，不删除 dataset/model workflow

Raymobtime s008 的本体价值是数据审计、cache 构建、current snapshot dataset、模型、单任务/多任务 objective 和指标输出。失衡诊断只是在这些 run 之上的解释层，因此删除 `kd-sensing-raymobtime-analysis` 和 `src/kd_sensing/diagnostics/raymobtime_analysis.py`，但保留 `src/kd_sensing/preprocessing/raymobtime_s008*`、`src/kd_sensing/models/raymobtime_s008.py` 和常规训练/评估配置。

替代方案是删除 Raymobtime s008 全部能力。该方案超出本次“退役失衡研究表面积”的范围，也会破坏已有 dataset/runtime contract。

### 4. Viewer 删除 Complementarity Explorer，保留 manifest viewer

`tools/visualization/gradio_multimodal_viewer.py` 应移除 complementarity import、CLI 参数、数据加载、Tab、回调和状态字段。保留 raw modalities、processed modalities、diagnostics、prediction distribution、manifest IO 和 viewer performance 优化相关模块。

替代方案是在 viewer 中保留空的 Complementarity Tab。该方案继续暗示研究流程可用，也保留额外测试和 UI 状态维护成本，因此不采用。

### 5. OpenSpec 先退约束，再删代码

实现时先更新 active change specs，再删除代码和文档。否则当前 specs 仍要求 Conditional Utility、Complementarity、Phase 1.5、Raymobtime analysis 和 G2D result collection 存在，会让后续测试和 agent 行为回流。

替代方案是只删源码不动 OpenSpec。该方案会制造规格漂移，不符合仓库工作规则。

## Risks / Trade-offs

- [Risk] 删除研究脚本后，旧实验产物无法用仓库当前工具重新生成同名报告。→ Mitigation：文档明确这些研究入口已退役；本地 `outputs/` 不迁移，历史文件可作为静态产物保留在用户机器上。
- [Risk] 误删通用 subset 或 mask 能力会破坏 CRAF/MARF 和 Raymobtime 模型测试。→ Mitigation：删除前按调用边界区分 Conditional Utility 专属 registry 与通用 `force_modality_mask`、`evaluation.modality_subsets`。
- [Risk] G2D 删除范围不清会破坏 CSI hardening matrix。→ Mitigation：只删除失衡汇总脚本与文档引用，保留 G2D distiller、SMP、training extension、diagnostics 和配置 overlay。
- [Risk] 架构边界测试和 docs inventory 会因入口删除而失败。→ Mitigation：同步更新 allowlist、project surface inventory、README 和 OpenSpec project-architecture。

## Migration Plan

1. 更新 OpenSpec delta，移除研究流程合同并修改保留能力合同。
2. 删除 Conditional Utility、Complementarity、Phase 1.5、Raymobtime imbalance analysis 和 G2D imbalance collector 的源码、脚本、配置与测试。
3. 从 Gradio viewer 删除 Complementarity Explorer 参数、Tab、回调和 helper import。
4. 更新 `pyproject.toml`、架构边界测试、脚本 allowlist 和 docs inventory。
5. 更新 README、`docs/experiment_matrix.md`、`docs/research_notes.md` 和 `tools/visualization/README.md`，去掉已退役研究入口。
6. 运行定向验证，再运行全量回归。

Rollback 策略：若实现后发现某个研究工具仍需保留，优先恢复对应文件和 spec delta 中的需求；本变更不修改本地输出产物，因此 rollback 不涉及数据迁移。

## Open Questions

- 是否需要保留一个文档页记录“模态失衡研究线已退役”的历史说明？默认不新增，以避免继续维护该主题。
- `scripts/eval_modality_subsets.py`、`scripts/eval_modality_perturbation.py` 和 `scripts/debug_eval_consistency.py` 是否算研究遗留？本方案默认保留，因为它们是通用调试工具，不依赖 Conditional Utility 或互补 case schema。
