## Context

科研项目最危险的漂移不是代码崩溃，而是 claim 状态漂移：mock/smoke、upper-bound、historical ablation 或 not_comparable 被误读成正式结果。当前 registry 已有状态体系，但需要把缺失字段和升级条件变成可执行报告。

## Goals / Non-Goals

**Goals:**

- 让每个 claim 的缺口可机器报告。
- 让每个 run 形成可追溯 run card。
- 让 paper export 默认拒绝不合格主表行。
- 给 agent 一个“下一步该补什么证据”的入口。

**Non-Goals:**

- 不自动把 claim 从 pending 升级为正式结论。
- 不提交真实 metrics、checkpoint、figures 或 cache。
- 不替代人工论文判断。

## Decisions

1. Claim doctor 输出缺失字段和 next actions。
   - 理由：比单纯显示 pending 更能指导下一步。
   - 备选：继续人工读 registry；容易漏 provenance。

2. Run card 作为 ignored output artifact，不直接写源码。
   - 理由：run card 包含本地产物路径和 metrics，默认不应提交。
   - 备选：直接写 docs；会把未审阅结果带入源码。

3. Paper export gate 默认 hard exclude 不合格状态。
   - 理由：论文表格比 dashboard 更需要保守。
   - 备选：导出后人工删除；容易出错。

## Risks / Trade-offs

- [Risk] Doctor 对历史 claim 报告过多噪声。→ Mitigation: 按 status 和 line_id filter，默认关注 pending/upgradable。
- [Risk] Run card schema 变复杂。→ Mitigation: 先覆盖 command/config/split/checkpoint/metrics/claim candidate 最小字段。
- [Risk] Dashboard 被误认为正式结果。→ Mitigation: 所有候选保持 `candidate_only=true` 或明确 status。

## Migration Plan

- 先扩展 claim doctor 和 paper gate tests。
- 再生成 run card artifact。
- 最后汇总到 research dashboard。

## Open Questions

- Run card 是否需要稳定 JSON schema 版本，建议实现阶段加入 `schema_version`。
