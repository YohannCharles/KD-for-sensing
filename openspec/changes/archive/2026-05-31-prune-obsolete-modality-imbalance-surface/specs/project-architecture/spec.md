## ADDED Requirements

### Requirement: 退役旧模态诊断脚本入口
项目 MUST 不再把模态失衡时期的独立模态子集和模态扰动研究脚本作为长期维护入口。通用模态 subset、mask 或 perturbation 调试能力如需保留，MUST 通过包内 CLI、配置化 evaluation pass、LOSO summary 或明确的内部 helper 承载，并 MUST 在脚本 allowlist 和项目表面积 inventory 中体现当前边界。

#### Scenario: 脚本入口清单不包含旧诊断脚本
- **WHEN** 开发者运行架构边界测试检查 `scripts/` 与 `tools/` 入口清单
- **THEN** `scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py` MUST 不再作为允许的长期入口存在
- **AND** 测试 MUST 继续允许当前保留的 thin CLI alias、dataset preparation、viewer 和 MMW/HiST-Beam orchestration 入口

#### Scenario: 通用 subset 能力不被误删
- **WHEN** evaluation 配置启用 `evaluation.modality_subsets`
- **THEN** 系统 MUST 继续能在共享 evaluation pass 中计算配置化 subset metrics
- **AND** 该能力 MUST 不依赖被退役的独立研究脚本

### Requirement: 表面积 inventory 跟随当前主线
项目 surface inventory MUST 将当前推荐入口描述为少样本跨场景、MMW/HiST-Beam、Raymobtime、CSI hardening、viewer manifest 和通用训练评估能力。已退役的模态失衡诊断脚本、KD virtual alias 扩展和不再推荐的研究路线 MUST 不作为新入口或健康检查要求出现。

#### Scenario: inventory 删除旧研究入口
- **WHEN** 开发者阅读 `docs/project_surface_inventory.md`
- **THEN** 文档 MUST 不再把旧模态子集/扰动诊断脚本列为长期维护 research diagnostic 入口
- **AND** 文档 MUST 保留本地产物边界说明，不要求删除或迁移历史 `outputs/`、`logs/` 或 `dataset/`
