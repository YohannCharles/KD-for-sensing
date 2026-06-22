## Context

IEEE WCL 2025 的 `Robust Multimodal Beam Prediction With Missing Modality` 是缺失模态 beam prediction 的直接相关对照，但目前不能假设官方代码、权重或完整训练 recipe 可用。本 change 需要先建立 source-audit 和 claim-status 边界，再决定使用 official-code reproduction 还是 paper-aligned local substitute。

该 change 与 AMBER-lite 分工不同：AMBER-lite 是本仓库最小强对照；WCL 2025 reproduction 是论文对齐实验，必须显式记录与论文的匹配程度和不可复现原因。

## Goals / Non-Goals

**Goals:**

- 审计论文、代码、权重、数据集、模态、split、metric、训练流程和缺失模态设置。
- 在 official artifacts 可用时包装官方复现流程。
- 在 official artifacts 不可用时实现 paper-aligned local substitute，并记录 deviation。
- 输出 condition-level metrics、strict comparability、claim status 和 provenance。

**Non-Goals:**

- 不在未确认官方 artifacts 时声称 official reproduction。
- 不把 WCL 2025 local substitute 与 AMBER-lite 混成同一模型。
- 不提交外部源码、checkpoint、dataset 或运行产物。
- 不恢复 retired KD/HiST/residual/BGAM 路线。

## Decisions

1. source audit 是第一阶段硬门槛。
   - manifest 必须记录论文元数据、代码 URL、source commit、license/availability、checkpoint availability、dataset/split/metric 匹配情况。
   - 缺少关键 artifact 时，claim status 只能是 `local_substitute`、`blocked`、`pending` 或 `not_comparable`。

2. official-code 与 local-substitute 分支共用 summary schema。
   - official 分支包装外部代码运行或导入其预测/metrics。
   - local-substitute 分支使用本仓库 component/workflow 实现论文描述的缺失模态模型。
   - 两者都输出同一 provenance 和 condition metrics 字段。

3. 模型实现优先可组合组件。
   - 若论文结构可表达为 per-modality encoders + missing-modality fusion core，则走 `modular_sequence` component baseline。
   - 只有论文需要不可拆的完整 forward 结构时，才在 design 更新中记录 whole-model exception 理由。

4. 评估与 claim 分离。
   - condition-level metrics 可以输出为 local evidence。
   - 只有 official artifacts、strict comparable protocol 和真实 run provenance 完整时，才允许进入 official claim。

## Risks / Trade-offs

- [Risk] 论文细节不足以完全复现。→ Mitigation: source-audit manifest 记录 missing details，并将结果标记为 local substitute。
- [Risk] local substitute 与 AMBER-lite 重复。→ Mitigation: WCL 2025 配置和 metadata 必须记录 paper alignment/deviation，不作为 generic lite baseline。
- [Risk] 官方代码依赖或 license 不适合 vendoring。→ Mitigation: 外部源码只作为本地路径或 git URL 引用，不提交到仓库。
- [Risk] strict protocol 不匹配当前 P0-P5 heatmap。→ Mitigation: comparability mismatch 阻止 strict ranking，只输出 external/local reference。
