## Context

主线演进记录显示：Scene31-34 已升级为缺失模态论文主实验，主方法候选冻结为 prototype + random subset exposure；JEPA predictive robustness 仍处于 smoke/pending，需要真实 benchmark 才能形成 claim。此阶段的价值在于补齐可比较证据，而不是继续引入新模块。

## Goals / Non-Goals

**Goals:**

- 给 Scene31-34 final evidence 建立明确 checklist 和 runner/summary 验收。
- 给 JEPA real benchmark 建立 promotion gate。
- 将 evidence completion 与 claim registry/paper export 连接起来。

**Non-Goals:**

- 不新增缺失模态模型结构。
- 不把 smoke 或 synthetic benchmark 提升为真实 claim。
- 不恢复 retired research lines。

## Decisions

1. Scene31-34 优先完成 evidence matrix，而不是继续搜索方法。
   - 理由：已有候选已足够，论文结论需要多 seed、classifier、external 和 compute 证据。
   - 备选：继续加 reliability/PatternFiLM 模块；会扩大范围且削弱主线清晰度。

2. JEPA real benchmark 必须从 audited manifest 启动。
   - 理由：shortcut/predictive claim 对 comparability 极敏感。
   - 备选：复用 smoke manifest；只能验证 schema，不能支持结论。

3. Claim 升级由 evidence gate 触发人工审阅。
   - 理由：避免自动把 pending 变正式。

## Risks / Trade-offs

- [Risk] Scene31-34 真实训练成本高。→ Mitigation: tasks 区分 generator/summary smoke、py_compile 和长训本地步骤。
- [Risk] JEPA checkpoint 路径本地不可用。→ Mitigation: real manifest 缺路径时输出 unavailable/not_comparable，不伪造结果。
- [Risk] 主线冻结让一些候选暂缓。→ Mitigation: reliability/PatternFiLM 保持 ablation/pending，不删除历史上下文。

## Migration Plan

- 先补 checklist、doctor 和 summary smoke。
- 再运行本地 long-run 或消费已有 outputs。
- 最后通过 claim doctor/paper export gate 升级或保持 pending。

## Open Questions

- Scene31-34 external-lite 是否只要求 seed1 maskfix，还是后续扩到 n=3，应由本地结果和 compute budget 决定。
