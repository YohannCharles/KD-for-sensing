## Context

Multimodal-NF 已支持本地数据布局审计、HDF5 index、flat dict sample、codebook metadata、near-field beam target、LOS/link 辅助任务和 fusion/multitask 配置。当前问题主要在解释层：archived spec 仍有 TBD Purpose，runtime metadata 中有早期 `future_near_field_beam_prediction` 命名，而当前实验已经覆盖 `near_field_beam_selection`、`current_los_classification`、`current_link_quality` 和 `selection_multitask`。

本变更应保持训练算法和配置入口稳定，只整理“怎么描述一个 Multimodal-NF run”。

## Goals / Non-Goals

**Goals:**
- 补齐 Multimodal-NF 和 dataset runtime specs 的真实 Purpose。
- 统一 Multimodal-NF runtime metadata，按 objective 表达 task semantics、target schema、codebook 和 enabled targets。
- 增加运行产物一致性检查，提前发现 `num_classes`、codebook shape、objective、modalities、heads 不一致。
- 更新文档，让用户能区分 dataset smoke、单任务、multitask 和 fusion run。

**Non-Goals:**
- 不改变 Multimodal-NF 原始数据布局、HDF5 index 格式或 codebook flatten 规则。
- 不重命名已有配置路径，不删除本地输出。
- 不改变 loss、metric 计算、模型结构或 checkpoint 选择策略。
- 不引入新的数据下载或迁移流程。

## Decisions

1. 以 objective metadata 作为 runtime 语义来源。
   - `near_field_beam_selection` 记录 near-field codebook class、Top-K triplet metadata 和 flattened target schema。
   - `current_los_classification` 记录 LOS binary classification target。
   - `current_link_quality` 记录 link quality regression target。
   - `selection_multitask` 记录 beam、LOS、link 三类 target 与 loss/metric 字段。

2. Dataset runtime metadata 保留 dataset family 与 target schema 双层信息。
   - dataset family 表示 `multimodal_nf`、storage kind、city split、input profiles。
   - target schema 表示当前 objective 实际消费的 label/auxiliary targets。
   - 这样避免把“数据集能力”误读成“当前 run 主任务”。

3. 产物一致性检查做成轻量校验。
   - 配置加载或启动阶段检查 codebook `num_beam_classes` 与模型 head `num_classes` 是否一致。
   - 写出 final config/startup summary 时记录 objective、modalities、heads 和 target schema。
   - 对历史输出不做迁移，只保证新 run 更一致。

4. 文档整理和 spec Purpose 修正与代码一起完成。
   - 这是小但重要的维护工作；当前 `project-architecture` 已要求 archived TBD Purpose 不应长期保留。

## Risks / Trade-offs

- [Risk] 更严格的一致性检查可能暴露历史配置中的隐性矛盾。→ Mitigation：错误信息给出配置字段和实际解析值，必要时先在 focused tests 中固定预期。
- [Risk] runtime 字段改名可能影响已有分析脚本。→ Mitigation：新增规范字段，保留旧字段或在同一 metadata 中提供兼容别名；不删除现有公开字段。
- [Risk] “current” 与 “near-field” 命名容易混淆。→ Mitigation：文档和 runtime 明确区分 Raymobtime current snapshot 与 Multimodal-NF near-field codebook target。

## Migration Plan

1. 补齐 OpenSpec Purpose 和需求 delta。
2. 调整 runtime metadata helper，新增规范字段并保留兼容字段。
3. 增加配置/启动一致性校验。
4. 更新 README/docs 中 Multimodal-NF 说明。
5. 跑 Multimodal-NF focused tests 和相关 objective tests。

回滚策略：保留旧 runtime 字段，关闭新增一致性校验或降级为 warning，不影响训练主路径。

## Open Questions

- `near_field_beam_selection` 是否应长期替代当前部分配置里的 `beam` 命名？本 change 只澄清，不强制批量重命名。
- 是否需要为 Multimodal-NF 单独增加 experiment matrix 汇总文档？可在本 change 中先补 README/docs 小节。
