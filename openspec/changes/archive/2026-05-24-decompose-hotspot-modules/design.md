## Context

项目已经建立了 `src/kd_sensing` 包结构、轻量导入边界、component registry、canonical config、objective metadata、viewer manifest 和多数据集 runtime 契约。当前剩余风险不是缺少模块边界，而是若干热点文件仍承担多个职责：配置/指标 schema、runtime metadata、IO、writer、validation 和业务分支混在一起。

这些热点模块短期可工作，但继续扩展会增加回归面。拆分需要保护公开入口、配置路径、artifact schema 和测试稳定性。

## Goals / Non-Goals

**Goals:**
- 制定并实现一批低风险热点模块拆分。
- 保持现有公开 import、CLI、配置路径和 artifact 字段兼容。
- 增加架构边界测试，防止新的二级兼容聚合层和重依赖 eager import 回流。
- 让后续 objective、Multimodal-NF、viewer 和 DeepVerse 相关功能更容易局部修改。

**Non-Goals:**
- 不删除用户可见 CLI 或配置入口。
- 不改变训练、评估、预处理、viewer manifest 的输出语义。
- 不做大规模命名迁移或旧 artifact 重写。
- 不在同一批次重构模型算法、dataset 数据语义或 loss/metric 数学定义。

## Decisions

1. 分层拆分顺序：schema/constants → pure helpers → readers/writers → orchestration。
   - 先移动无副作用代码，减少行为变化风险。
   - orchestration 函数最后处理，并保留原公开入口调用新窄模块。

2. 保留公开 facade，但禁止新增内部二级兼容依赖。
   - 现有用户可能从旧公开模块 import；这些路径可以继续 re-export。
   - 新内部代码必须直接使用窄模块。
   - 架构测试扫描内部 import，防止又长出 `_legacy`、`_builders_impl` 式聚合。

3. 每个热点模块单独验收。
   - `objective_metadata`：目标 registry、alias、history/TensorBoard schema、validation helper 拆开。
   - Multimodal-NF preprocessing：path/audit/index/split/codebook/HDF5 inspection 拆开。
   - viewer manifest：manifest schema、prediction merge、cache metadata、writer/export 拆开。
   - DeepVerse label builder：scene metadata、target derivation、split、sanity check、writer 拆开。

4. 用 focused tests 锁定行为。
   - 每次拆分都先跑对应 focused tests。
   - 涉及公开入口或 artifact 字段时补 snapshot/field presence 断言。

## Risks / Trade-offs

- [Risk] 拆分产生大量小文件，导航成本上升。→ Mitigation：每个子包保留 README 或清晰 `__all__`，公开入口继续聚合用户需要的符号。
- [Risk] Re-export facade 被新代码误用。→ Mitigation：架构边界测试拒绝内部代码引用指定 facade。
- [Risk] 移动代码时无意改变 artifact 字段。→ Mitigation：增加 train_log/final_config/metrics/viewer manifest 字段断言。
- [Risk] 一次拆太多难以定位问题。→ Mitigation：任务按热点区域拆分，可分批实施和验证。

## Migration Plan

1. 添加 hotspot inventory 和目标模块图。
2. 拆 `objective_metadata` 的纯表和 helper。
3. 拆 Multimodal-NF preprocessing/common helper。
4. 拆 viewer manifest 相关 schema/merge/writer。
5. 拆 DeepVerse label builder。
6. 扩展架构边界测试和文档。

回滚策略：保留原公开模块为 facade，可将某个 facade 临时改回本地实现；不需要迁移用户配置或本地 outputs。

## Open Questions

- 是否给 `engine/objectives/` 新建子包，还是保持在 `engine/` 平铺若干 `objective_*` 模块？建议实现时根据 import 边界测试决定。
- DeepVerse label builder 是否在本批次完成，还是先只做 inventory 和测试护栏？可按时间拆到后续 change。
