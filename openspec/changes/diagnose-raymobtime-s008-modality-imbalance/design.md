## Context

当前 s008 seed42 结果呈现一个清晰但尚未闭环的现象：

- beam 单任务：`lidar` 与 `coord+image+lidar` 最强，`image` 明显弱。
- LOS/link 单任务：`coord+image+lidar` 与 `lidar` 同样占优。
- `selection_multitask`：LOS/link 继续变好，但 beam top1 明显低于 beam 单任务 LiDAR/CIL，甚至低于 coord beam 单任务。
- 已生成的模态失衡分析目录中，单任务性能表可用，但 gate、drop modality、gradient 和 LOS bucket 表目前只有表头，缺少内部证据。

这说明我们已经有“疑似多任务模态失衡”的外部指标证据，但还不能排除以下混杂：

- `val_selection_multitask_loss` early stopping 没有选择 beam 最优 epoch。
- link regression loss 尺度或权重压过 beam CE。
- 单 seed 偶然性。
- 多任务组合本身引入负迁移，而不一定是模态失衡。
- 诊断工具没有产出 gate/drop/gradient 数据，导致无法解释模型到底依赖了哪个模态。

## Goals / Non-Goals

**Goals:**

- 设计一套 s008 优先的实验矩阵，用于确认失衡是否稳定存在，并排除代码/参数混杂。
- 明确每组实验的主问题、控制变量、输出指标和判定标准。
- 补齐 gate、test-time modality drop、gradient/contribution、LOS bucket 等内部诊断证据。
- 给出 s009 作为第二阶段外部验证的进入条件，避免过早扩大变量空间。

**Non-Goals:**

- 不在本 change 中实现新的 s009 dataset 契约或迁移 s008 预处理逻辑到 s009。
- 不改变 Raymobtime s008 当前快照语义、模型输出形状、正式 metrics 命名或训练入口。
- 不把 checkpoint、cache、TensorBoard、训练日志或本地数据纳入源码变更。
- 不用 s009 的结果反向替代 s008 的失衡确认。

## Decisions

### Decision 1: 先确认 s008，再进入 s009

s009 只能回答“这种现象是否跨场景存在”，不能回答“s008 的现象是不是代码或参数问题”。因此第一阶段只在 s008 上做诊断闭环；当 s008 满足确认标准后，再把最小矩阵迁移到 s009。

备选方案是立即做 s009 横向比较。这个方案看似更快，但会同时引入数据契约、cache、label 对齐、split 策略和预处理支持差异，反而削弱因果判断。

### Decision 2: 固定数据 split，先变化训练 seed

第一轮多 seed 实验固定 Raymobtime s008 cache、split seed、portion 和输入样本集合，只变化 `experiment.seed`，建议使用 `42/7/123`。这样能先判断训练随机性是否解释 beam 退化。若三 seed 都显示相同趋势，再增加 split seed 作为稳健性补充。

### Decision 3: 把矩阵拆成四个层次

实验矩阵按证据强度从外到内排列：

```text
外部性能证据
  ├─ 单任务 × 单/融合模态 × 多 seed
  ├─ 多任务 task-combo / loss-weight / early-stop 消融
  │
内部机制证据
  ├─ gate mean by task / LOS bucket
  ├─ test-time modality drop delta
  ├─ gradient 或 contribution by task/modality
  └─ beam metrics by LOS bucket
```

外部性能只能说明“发生了负迁移”。内部机制用于说明“负迁移是否由模态/任务支配导致”。

### Decision 4: 多任务消融必须同时覆盖任务组合和 loss 尺度

核心多任务对照包括：

- `beam_only_multitask_model`: 使用同一 task-aware 模型和 CIL 输入，但只启用 beam loss。
- `beam+los`: 排除 link regression 对 beam 的影响。
- `beam+link`: 排除 LOS classification 对 beam 的影响。
- `beam+los+link original`: 当前配置权重，例如 `beam=1, los=0.5, link=0.2`。
- `beam+los+link equal`: 等权重，用于暴露尺度问题。
- `beam_heavy`: 提高 beam 权重或降低 los/link 权重，用于观察 beam 是否可恢复。

如果 beam 只在等权或 link 权重大时退化，但 beam-heavy 恢复到 CIL beam 单任务附近，则优先判为参数/loss 尺度问题。如果 beam-heavy 仍不能恢复，同时内部诊断显示 gate/drop/gradient 偏向非 beam 有效模态或特定辅助任务，则更支持模态失衡/任务冲突。

### Decision 5: checkpoint 选择作为独立消融

每个 multitask run 需要汇总至少三种 epoch 视角：

- best `val_selection_multitask_loss`
- best `val_beam_top1`
- best `val_link_mae`

如果训练日志显示 best beam epoch 存在但 early stopping 选择了别的 epoch，结论不能直接写成数据固有失衡。只有在 best beam 视角仍显著低于 beam 单任务对照时，才能继续推进到失衡判定。

### Decision 6: s009 只复刻最小矩阵

s009 阶段不复刻完整 s008 大矩阵。进入 s009 后只运行：

- 单任务：`lidar`、`coord+image+lidar` 的 beam/LOS/link。
- 多任务：original、beam-heavy、最可疑的 task-combo。
- 诊断：gate、drop、gradient 和 LOS bucket。

如果 s009 需要新增 dataset/preprocess 能力，应另开 change，而不是把 s009 兼容性塞进本诊断 change。

## Risks / Trade-offs

- [训练成本上升] 多 seed 和消融矩阵会增加运行量。缓解方式：先跑 seed42 的完整矩阵，再对关键对照扩展到 `7/123`。
- [结论被早停污染] `val_selection_multitask_loss` 可能不代表 beam 最优。缓解方式：所有 multitask run 都必须做 checkpoint/epoch 视角汇总。
- [loss 尺度混杂] link loss 未归一化或权重过大会主导优化。缓解方式：把 original、equal、beam-heavy 和 task-combo 消融列为必要矩阵。
- [诊断产物为空] 当前 gate/drop/grad 产物可能无法直接生成。缓解方式：先作为诊断可用性门禁；若工具缺能力，记录为需要实现的任务，再跑实验。
- [s009 变量过多] 过早引入 s009 会模糊 s008 结论。缓解方式：s009 必须等待 s008 判定报告完成后进入。

## Migration Plan

1. 复用现有 s008 cache 和 seed42 单任务结果，整理 baseline inventory。
2. 生成或整理 s008 诊断矩阵配置，所有命令使用 `conda run -n kd_mm_beam ...`。
3. 先完成 seed42 的 task-combo、loss-weight、checkpoint 选择消融。
4. 将关键对照扩展到 seed `7/123`。
5. 生成 s008 判定报告，明确结论属于“确认失衡”“参数问题优先”“证据不足”之一。
6. 只有 s008 结论为“确认失衡”或“高置信疑似失衡”时，才进入 s009 最小复刻矩阵。

## Open Questions

- 当前训练流程是否保存足够 checkpoint 支持 best-by-beam 与 best-by-link 的离线复评估。
- gate/drop/gradient 诊断为空是因为分析入口缺少 checkpoint、缺少诊断 hook，还是因为当前模型没有暴露所需 diagnostics。
- s009 本地数据是否已经满足与 s008 相同的 beam/LOS/link 标签和模态路径契约。
