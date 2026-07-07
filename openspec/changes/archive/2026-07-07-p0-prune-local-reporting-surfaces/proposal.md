## Why

本轮 ponytail-audit 显示，仓库当前最大的不必要代码量集中在本地报告、论文表格和一次性分析脚本，而不是训练核心或模型核心。`scripts/analysis/`、Scene31-34 final analysis、Scene31 单场景汇总和 apples-to-apples 复评入口合计已经超过一万行，并且很多脚本只服务某一轮结论导出、表格排版或展示材料。

这些脚本对实验研究早期很有用，但在结论已经沉淀到 docs、paper tables、claims 或正式 artifact 后，继续把每个临时入口都留在 current surface，会扩大维护面、增加测试和 inventory 噪声，并让后续协作者误以为所有脚本都仍是当前研究流程的一部分。

本 change 目标是 P0 级瘦身：优先处理最高代码量、最低复用价值的本地报告面。实现时必须先保留结论证据、更新 current specs/docs/inventory，再删除或合并脚本；不允许用新的兼容 wrapper 代替旧 wrapper。

## What Changes

- **BREAKING**：`scripts/analysis/` 下的一次性研究分析脚本不再作为 current source surface 保留；结论、输入路径和关键输出必须先迁入正式 docs、paper tables、claim notes 或 retained artifact 说明。
- Scene31-34 final analysis cluster 收敛为一个明确 owner 命令或 package-local helper；旧的 per-artifact 导出脚本在输出契约被覆盖后删除。
- Scene31 单场景汇总脚本收敛为一个参数化 summary owner；BC-next、P0 fresh eval、baseline pack、subset reliability、patternfilm、funnel、next-round 和 subset reference 通过显式 profile/group 参数表达。
- `scripts/reevaluate_apples_to_apples.py` 的 fresh evaluation 行为折回 package evaluate workflow 或 canonical eval helper；删除独立大型脚本或将其降为窄 documented recipe。
- 更新 README、docs、OpenSpec current specs、`docs/project_surface_inventory.md`、focused tests 和 surface guardrail，使被删除脚本不会被重新要求存在。

## Capabilities

### New Capabilities

- 无。本 change 只删除或合并现有本地报告和分析表面。

### Modified Capabilities

- `project-entrypoint-lifecycle`：增加本地报告脚本的生命周期规则，要求一次性脚本在结论沉淀后退出 current surface。
- `scenes31-34-main-missing-modality-workflow`：允许 Scene31-34 final analysis 从多个 per-artifact 脚本收敛到一个 owner，同时保持表格、图、结论和验证契约。
- `scene31-next-round-experiment-workflow`：允许 Scene31 next-round 相关 summary 由共享 Scene31 summary owner 产出，而不是保留多个同构脚本。
- `scene31-baseline-pack`：允许 baseline pack 汇总接入共享 Scene31 summary owner，并明确旧脚本删除后的输出保持要求。

## Impact

- 影响范围：`scripts/analysis/*.py`、Scene31-34 final analysis 脚本、Scene31 summary 脚本、`scripts/reevaluate_apples_to_apples.py`、README/docs/OpenSpec references、project surface inventory、focused tests 和 scripts guardrail。
- 不影响范围：训练数学语义、模型 forward、dataset split、canonical configs、package CLI 主入口、正式评估指标、本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和 `All_models/`。
- 兼容性：被删除的一次性脚本路径不再承诺可用；替代路径是 consolidated owner command、package evaluate workflow 或文档化的 retained artifact。
- 验证：OpenSpec strict validate、scripts surface doctor、相关 Scene31/Scene31-34 focused tests、CLI help/config smoke tests，以及对纸面表格或 claim notes 的字段级对照检查。
