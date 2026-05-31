## Why

项目主线已经从模态失衡解释与多模态蒸馏转向少样本跨场景自适应，旧的模态子集/扰动诊断脚本、KD virtual 配置面和 teacher-student 命名继续占据入口、配置、测试和文档表面积。现在需要第二轮收窄：保留当前可复用能力，但移除或降级不再服务 HiST-Beam、MMW LOSO、history-anchor 和 target adaptation 的旧研究入口。

## What Changes

- **BREAKING**：退役 `scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py` 作为长期维护脚本入口；通用模态子集评估应通过 `kd-sensing-evaluate`、配置化 `evaluation.modality_subsets` 或 LOSO/evaluation summary 暴露。
- **BREAKING**：停止由 canonical virtual config 生成新的 fusion `logits_kd` / `rkd` 配置路径；历史 KD 只保留显式实体配置或明确 legacy/baseline 路径，不再作为 canonical fusion mode 的默认表面积。
- 收窄 no-KD 配置：no-KD 主线配置不再携带 `temperature`、`alpha`、`rkd_*` 等 KD-only 超参；配置和文档把 `distillation.type: no_kd` 作为兼容开关而不是主线概念。
- 保留 legacy KD 算法本体和已跟踪单模态 `logits_kd` / `rkd` 配置，除非后续独立 change 明确删除全部历史 KD baseline。
- 保留 `evaluation.modality_subsets`、`force_modality_mask`、viewer manifest、Raymobtime s008、CSI hardening 和 HiST-Beam/MMW 当前主线能力。
- 标记 objective-aware occlusion/position/multitask 为保留但非当前 few-shot cross-scene 默认主线；本变更不删除这些任务，只要求 README/实验矩阵不把它们列为当前推荐主路线。
- 更新 README、docs、project surface inventory、架构边界测试和配置测试，使入口清单反映当前少样本跨场景主线。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 收窄脚本入口和源码表面积，退役旧模态子集/扰动研究脚本，要求通用能力通过包内 CLI 或配置化评估路径承载。
- `legacy-kd-isolation`: 进一步收窄 legacy KD 暴露面，停止把 KD virtual fusion alias 作为 canonical 配置主入口，并清理 no-KD 配置中的 KD-only 字段。
- `experiment-workflow`: 调整推荐 workflow 与健康检查，不再把模态失衡诊断脚本、KD virtual alias 或 objective-aware auxiliary tasks 作为当前 few-shot cross-scene 默认路线。
- `configurable-multimodal-fusion`: 调整 canonical fusion 生成规则，使默认 virtual config 聚焦 no-KD strong/lightweight/fusion 主线，legacy KD 不再自动扩展为所有模态组合的虚拟入口。

## Impact

- 影响脚本与入口：`scripts/eval_modality_subsets.py`、`scripts/eval_modality_perturbation.py`、`tests/test_architecture_boundaries.py`、`docs/project_surface_inventory.md`。
- 影响配置生成：`src/kd_sensing/config/canonical.py`、`src/kd_sensing/config/canonical_recipes/fusion.py`、配置加载测试和 KD lineage 测试。
- 影响显式配置：no-KD YAML 与 virtual no-KD 输出应移除 KD-only 超参；单模态 `configs/**/{logits_kd,rkd}.yaml` 可作为 legacy baseline 暂时保留。
- 影响文档：README、`docs/experiment_matrix.md`、`docs/extension_guide.md`、`docs/research_notes.md` 和 OpenSpec active specs。
- 不影响本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或已归档 OpenSpec 历史记录；不新增外部依赖。
