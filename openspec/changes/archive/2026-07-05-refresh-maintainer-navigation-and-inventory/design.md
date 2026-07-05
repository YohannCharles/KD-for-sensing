## Context

项目已经有比较完整的 OpenSpec、README、AI 导航和 project surface inventory。当前问题不是缺少治理，而是治理事实出现轻微漂移：架构边界测试发现 4 个 tracked `scripts/` 文件未登记，inventory 的规模基线仍停留在旧快照，AI 进入项目时需要读较长链路才能获得“当前主线、入口、退役边界、验证命令”的一屏判断。

本 change 是文档和静态护栏同步，不改变训练、评估、预处理、配置解析、模型构建或输出产物语义。

## Goals / Non-Goals

**Goals:**

- 让 `tests/test_architecture_boundaries.py` 对未分类脚本的检查重新通过。
- 刷新 inventory 的 tracked 文件规模和配置数量说明，避免后续维护者基于过期数字判断项目表面积。
- 在 `docs/agent_navigation.md` 顶部提供一屏摘要，降低 AI/维护者启动一次非平凡改动的上下文成本。
- 明确 Scene31 / Scene31-34 论文表格和结论导出脚本属于本地研究报告脚本，输出只能写 ignored `outputs/` 或显式本地路径。

**Non-Goals:**

- 不迁移 `scripts/` 中的大段报告逻辑到 `src/kd_sensing/diagnostics/`；这需要独立 change 和 focused tests。
- 不新增 package CLI、兼容 wrapper、root-level 训练入口或新的配置生成机制。
- 不调整任何模型、dataset、training loop、evaluation loop、checkpoint schema 或 runtime output layout。
- 不清理或移动 `outputs/`、`logs/`、`dataset/`、cache、checkpoint 等本地产物。

## Decisions

1. **用现有 inventory 作为脚本分类权威。**  
   备选方案是放宽架构边界测试或在测试中新增 allowlist。放宽测试会掩盖真实漂移，测试 allowlist 会制造第二份事实来源；因此只更新 `docs/project_surface_inventory.md`。

2. **AI 一屏摘要放在现有导航文档顶部。**  
   备选方案是新增 `docs/ai_quick_context.md`。新增文档会增加一个入口层；把摘要放在 `docs/agent_navigation.md` 顶部能复用现有 AGENTS/README 指向。

3. **刷新规模基线但保持解释性而非硬 KPI。**  
   规模数字只用于帮助判断趋势，不用于机械拆分。文档继续说明真正判断来自 owner 职责、public surface、退役边界和 focused validation。

4. **脚本逻辑迁移只登记为后续改进。**  
   4 个漏登记脚本合计约千行，确实有包内 owner 化价值；但本 change 的最小目标是恢复治理一致性。把迁移纳入本轮会扩大风险并牵涉输出 schema。

## Risks / Trade-offs

- **Risk: 规模数字再次过期。** → Mitigation: 文档写清统计口径和复核命令倾向，后续可在独立 change 中将基线生成脚本化。
- **Risk: AI 摘要和正文重复。** → Mitigation: 摘要只保留决策入口，不复制完整任务路由和 OpenSpec requirement。
- **Risk: 登记脚本后被误认为推荐入口。** → Mitigation: inventory 明确标为 research diagnostic / local reporting，说明不是 package CLI，输出限定 ignored runtime roots。
- **Risk: 未做脚本 owner 迁移导致后续仍有跳转成本。** → Mitigation: 在 inventory 和任务说明中保留后续收敛方向，但本轮不引入行为变更。
