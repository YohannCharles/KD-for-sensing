## Context

当前 inventory 已经说明哪些 scripts/configs/hotspots 是 current、supporting、local/manual 或 retired guard。但这些规则主要是文档和测试断言，缺少一个面向维护者的快速报告工具。Doctor 的目标是把“读文档判断”变成“读文档之前先得到候选问题列表”。

## Goals / Non-Goals

**Goals:**

- 给 scripts/configs/hotspots 提供只读诊断报告。
- 输出可定位的文件、分类、原因、建议动作和验证命令。
- 支持 agent 在非平凡改动前快速发现未分类入口、失效 config 引用和热点扩大。

**Non-Goals:**

- 不自动删除 scripts/configs。
- 不自动生成真实训练配置并提交。
- 不用行数单独决定拆分或合并。

## Decisions

1. Doctor 默认只读，输出 JSON/Markdown。
   - 理由：符合本地产物边界和 OpenSpec 治理方式。
   - 备选：自动修复；风险过高，容易删除本地研究入口。

2. Scripts doctor 从 inventory/docs/specs/pyproject 推导生命周期。
   - 理由：避免测试维护一份重复 allowlist。
   - 备选：单独 YAML allowlist；容易漂移。

3. Config doctor 先做分类和引用检查，再规划 recipe migration。
   - 理由：很多 YAML 有实验差异，不能仅因重复就删除。
   - 备选：立即把所有 YAML 生成化；风险高且影响复现实验。

4. Hotspot doctor 输出 next-touch 建议，不直接重构。
   - 理由：热点处理必须结合 owner、public surface 和 focused tests。

## Risks / Trade-offs

- [Risk] Doctor 报告噪声太大。→ Mitigation: 输出 severity 和 lifecycle，默认只失败高风险项。
- [Risk] Doctor 变成新权威。→ Mitigation: 报告必须引用 inventory/spec/pyproject 来源。
- [Risk] Recipe migration 丢失实验语义。→ Mitigation: 只有无损可生成且有 tests 时才迁移。

## Migration Plan

- 先实现 scripts/configs 只读报告。
- 再把报告接入 architecture boundary 或 verify。
- 最后按 family 推进 recipe migration。

## Open Questions

- Doctor 是否作为 package CLI 暴露，还是仅作为开发脚本，需要实现阶段结合入口生命周期决定。
