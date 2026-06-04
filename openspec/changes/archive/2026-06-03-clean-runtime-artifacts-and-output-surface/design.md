## Context

当前项目已经通过 `src/kd_sensing/` 包结构、`.gitignore` 和 `tests/test_architecture_boundaries.py` 建立了源码与本地产物边界；`project-architecture` 也要求删除旧失败实验产物前先生成可审计清单。实际工作区盘点显示，本地数据和运行产物规模继续增长：`dataset/` 约 244G，`outputs/` 约 5.0G，`logs/` 约 211M，`outputs/` 内 `.pth` checkpoint 约 4.7G；其中 `outputs/other/` 由 MMW shell wrapper 默认输出产生，命名语义弱且单独约 1.2G。

这次 change 不解决实验结果好坏，也不替用户判断哪些科学结论可丢弃；它只把“能不能删、为什么删、删前记录什么、以后默认写到哪里”变成项目内可测试的工作流。

## Goals / Non-Goals

**Goals:**

- 提供只读清理候选扫描和 manifest 生成能力，默认不删除任何文件。
- 将清理候选按风险和语义分类：缓存/bytecode、短生命周期 plan/debug 产物、失败/partial/stale run、重复 checkpoint、语义不清的历史输出目录。
- 明确保护边界：`dataset/`、`All_models/`、源码、配置、文档、OpenSpec artifacts、已跟踪文件和未匹配的活跃实验默认不可删。
- 扩展 run index，让清理能力复用现有运行发现、状态推断、日志关联和 checkpoint 摘要。
- 将 MMW modal15 shell 默认输出从 `outputs/other/` 收敛到实验族专属目录，减少后续产物混淆。
- 为 checkpoint retention 建立可配置策略，支持保留复现必需文件并将可恢复/临时 checkpoint 列为候选。

**Non-Goals:**

- 不自动删除真实数据、训练输出、日志、checkpoint 或 cache。
- 不移动或重写历史 `outputs/`、`logs/` 目录。
- 不改变训练指标、模型结构、数据集 split 或实验结论。
- 不把本地产物纳入版本控制。
- 不替代人工确认；删除动作必须显式触发。

## Decisions

### 决策 1：采用两阶段清理流程

清理流程分为 `manifest` 和 `apply` 两阶段。第一阶段只扫描并写出 JSON manifest；第二阶段必须读取 manifest，并要求显式参数确认才执行删除。

理由：当前项目产物既包含临时 debug，也包含可复现实验 checkpoint。直接按路径或年龄删除风险太高，manifest 能让用户审阅候选路径、大小、原因和保护状态。

备选方案：只提供 shell `find ... -delete` 命令。放弃原因是不可审计，且容易误删活跃实验。

### 决策 2：以 run index 为清理扫描的底座

清理候选扫描复用 `kd_sensing.diagnostics.run_index` 的 run discovery、状态推断、log 关联和 checkpoint 摘要，并在其上补充大小统计、候选理由和保护规则。

理由：run index 已经是只读能力，且规格明确要求不修改产物。继续扩展它比新建一套路径扫描逻辑更容易保持一致。

备选方案：新增完全独立的 cleanup scanner。放弃原因是会重复实现 run 状态判断，后续容易和 run index 漂移。

### 决策 3：清理 manifest 使用稳定 schema

manifest 至少包含：生成时间、项目根、扫描根、候选列表、保护列表、总大小、候选大小、规则版本、dry-run 命令参数。每个候选记录路径、产物类型、大小、mtime、匹配规则、风险等级、是否已跟踪、是否受保护、删除动作建议。

理由：manifest 后续可以被测试、审阅、归档，也可以作为删除阶段的唯一输入。

备选方案：只输出人类可读 markdown。放弃原因是无法稳定驱动 apply 阶段和测试。

### 决策 4：默认保护优先于匹配规则

即使命中 `_debug`、cache 或 checkpoint 模式，只要路径属于受保护根、被 git 跟踪、位于 OpenSpec/源码/配置/文档内，或 run index 判断为 running/waiting，系统都必须标记为 protected，不得进入可删除候选。

理由：保护规则是最后一道防线，应优先于所有便利性清理。

### 决策 5：收敛 `outputs/other/` 而不是静默迁移

MMW shell wrapper 的默认 `OUTPUT_ROOT` 改为实验族目录，例如 `outputs/mmw_sunny_modal15/<horizon_tag>/`。历史 `outputs/other/` 不自动移动；它只作为 manifest 候选或人工保留对象出现。

理由：移动历史产物会破坏已有日志、文档和手工路径引用。默认路径收口能阻止新债务继续增长。

### 决策 6：checkpoint retention 默认保守

默认保留 `best.pth`、`best_top1.pth`、sidecar metadata、`metrics.json`、`final_config.yaml`、`resolved_config.yaml`、`startup_summary.json` 和必要 normalization artifacts。`last.pth`、重复 probe checkpoint、失败 run 的临时 checkpoint 可进入候选，但必须带有 run 状态和保留原因。

理由：checkpoint 是 `outputs/` 的主要空间来源，但也是复现关键。默认策略应先提供候选，而不是自动瘦身。

## Risks / Trade-offs

- 清理规则误判活跃实验 → manifest 阶段必须标记 running/waiting/stale 状态，apply 阶段再次检查路径存在、受保护状态和可选 mtime。
- manifest 太长难审阅 → 同时输出 JSON 和摘要表，按风险等级、目录和产物类型聚合大小。
- checkpoint retention 过于保守释放空间有限 → 先保证不误删，再允许用户显式选择更激进 profile。
- `outputs/other/` 改默认路径后历史脚本说明漂移 → 同步更新 shell help、inventory 和架构边界测试。
- 与本地未提交实验产物混淆 → 流程只处理 `.gitignore` 覆盖的本地产物，已跟踪文件一律 protected。

## Migration Plan

1. 增加清理 manifest 数据模型和只读扫描 helper，复用 run index。
2. 扩展 run index 输出大小、checkpoint retention 摘要和清理候选理由。
3. 增加 CLI dry-run 入口，默认只写 manifest。
4. 增加显式 apply 入口，要求传入 manifest 和确认参数。
5. 修改 MMW modal15 shell 默认输出根和帮助文本。
6. 更新 docs inventory、README/扩展指南中的产物边界说明。
7. 增加架构边界测试和清理 manifest 单元测试。

回滚策略：如果清理扫描误报，保留 manifest 只读能力，禁用 apply 入口或将删除阶段保留在文档外；MMW 输出路径可通过 `OUTPUT_ROOT` 环境变量临时恢复。

## Open Questions

- 是否需要在第一次实现中加入压缩归档动作，还是只支持删除候选 manifest？
- checkpoint retention 是否需要按实验族提供预设 profile，例如 `conservative`、`failed-runs-only`、`drop-last-checkpoints`？
- `outputs/analysis/` 中的图表和 summary 是否应单独拥有比训练 checkpoint 更短的默认保留策略？
