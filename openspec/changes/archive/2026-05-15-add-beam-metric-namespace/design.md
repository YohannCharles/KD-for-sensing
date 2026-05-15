## Context

当前 `src/kd_sensing/engine/trainer.py::_write_tensorboard_scalars` 每个 epoch 都写入 `accuracy/train`、`accuracy/val`、`accuracy/val_atop3`、`accuracy/val_atop5` 和 `dba/val_adba`。这些值来自 beam logits 与 `target_beam` 的诊断计算，即使 `experiment.objective` 是 `occlusion` 或 `position`，模型仍可能产出 beam logits，因此通用 `accuracy/*` 卡片会显示非 beam 单任务 run。

`occlusion` 和 `position` 已经采用 task namespace，并且只在对应指标真实可用时写入，例如 `occlusion/accuracy`、`occlusion/blocked_f1`、`position/rmse`、`position/mae`。beam 需要同样的一等指标命名空间，但不能把 occlusion-only 或 position-only run 中的诊断性 beam accuracy 继续写进去。

## Goals / Non-Goals

**Goals:**

- 为 beam 预测提供稳定的 canonical TensorBoard tag：`beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5`、`beam/val_adba`。
- 只在 beam objective 活跃时写入 `beam/*`：`beam` 和 `multitask` 可以写，`occlusion` 与 `position` 单任务不写。
- 让 early stopping metric alias 同时支持 `beam/*` 与历史 `accuracy/*`、`dba/*` 名称。
- 保留 `history`、`train_log.json`、`training_outputs.npz` 里的既有内部 key，减少对分析脚本的影响。

**Non-Goals:**

- 不新增独立 beam 模型、beam head 或训练脚本。
- 不改变 beam、occlusion、position 的 target 生成、loss 计算或 validator 诊断指标。
- 不重写历史 outputs 目录中的 TensorBoard event 文件。

## Decisions

### Decision 1: 新增 beam namespace，而不是新增 beam 任务实现

实现应在 TensorBoard writer 层新增 canonical tag 映射：

| TensorBoard tag | 内部 history key | 写入条件 |
| --- | --- | --- |
| `beam/accuracy_train` | `train_acc` | beam active |
| `beam/accuracy_val` | `val_acc` | beam active |
| `beam/val_atop3` | `val_atop3` | beam active |
| `beam/val_atop5` | `val_atop5` | beam active |
| `beam/val_adba` | `val_adba` | beam active |

Rationale: 当前 beam 指标已经存在，问题是 tag 命名和 active objective 过滤不完整。复用现有指标能避免重复计算和模型接口变化。

Alternatives considered:

- 直接复制 occlusion 的实现新增一个 beam auxiliary metric 模块：会制造重复的 beam 评估路径，且不能解决 occlusion/position 单任务诊断值混入的问题。
- 只改文档要求用户过滤 run：成本低，但 TensorBoard 默认卡片仍然误导。

### Decision 2: 使用 objective 判断 beam 是否 active

`beam_active` 由 `resolve_prediction_objective(cfg)` 或训练循环中已解析的 `objective` 决定。`objective in {"beam", "multitask"}` 时 beam 指标是 active；`objective in {"occlusion", "position"}` 时 beam 指标只是诊断，不得写入 `beam/*`。

Rationale: `available_metrics` 只能说明数值可计算，不能说明它是当前实验目标。occlusion-only/position-only run 中 beam logits 仍可计算，但不应出现在 beam 任务精度图中。

Alternatives considered:

- 只根据 `val_acc` 是否 finite 写入：会继续混入 occlusion/position 单任务。
- 只在 `objective == "beam"` 写入：会漏掉 multitask 中真实参与优化的 beam 分任务指标。

### Decision 3: 通用 `accuracy/*` tag 作为迁移兼容路径

新增配置项 `output.tensorboard.legacy_accuracy_tags` 控制是否继续写历史通用 tag。新推荐行为是不依赖 `accuracy/*` 分组；需要旧 dashboard 的用户可以显式启用兼容写入。内部 history key 和 JSON/NPZ 输出保持不变。

Rationale: 彻底删除历史 TensorBoard tag 会影响已有 dashboard；默认继续把它作为主入口又会保留混图问题。配置开关给迁移留出口，同时让新图表以 `beam/*` 为准。

Alternatives considered:

- 默认同时写 `beam/*` 和 `accuracy/*`：兼容性最好，但用户仍会看到混杂的旧 accuracy 卡片。
- 立即删除所有 `accuracy/*` 写入：最干净，但对只读取 TensorBoard event tag 的脚本不友好。

### Decision 4: 早停别名跟随 canonical tag

`_EARLY_STOPPING_METRIC_ALIASES` 应新增 `beam/accuracy_val`、`beam/val_top1`、`beam/val_adba` 等别名，并继续保留 `accuracy/val`、`dba/val_adba`。这只影响配置解析，不改变内部 early stopping key，仍然解析为 `val_acc` 或 `val_adba`。

Rationale: TensorBoard tag 和用户可配置 metric 名称应可互相对应，避免用户看到 `beam/val_adba` 后无法在配置中使用。

## Risks / Trade-offs

- [Risk] 旧 TensorBoard dashboard 依赖 `accuracy/val`。Mitigation: 提供 `output.tensorboard.legacy_accuracy_tags` 恢复旧 tag，并在 README 记录迁移方式。
- [Risk] 历史 runs 仍包含混杂的 `accuracy/*` 曲线。Mitigation: 文档说明该变更只影响新训练，历史图应按 run 的 `experiment.objective` 解释。
- [Risk] multitask run 出现在 `beam/*` 中可能被误解为 beam-only。Mitigation: 这是 active beam 分任务指标，README 中说明 `beam/*` 包含 beam-only 与 multitask 中的 beam 分量；纯单任务对比时可按 run name 过滤。

## Migration Plan

1. 调整 TensorBoard scalar writer，让它接收 objective 或 objective metadata，并按 `beam_active` 写入 `beam/*`。
2. 增加 `output.tensorboard.legacy_accuracy_tags` 解析和默认值，按配置决定是否写历史 `accuracy/*`、`dba/val_adba`。
3. 扩展 early stopping alias，支持 `beam/*` 名称。
4. 更新 TensorBoard regression tests，覆盖 beam objective 写 `beam/*`、occlusion/position 单任务不写 `beam/*`、legacy 开关恢复旧 tag。
5. 更新 README 的 TensorBoard tag 说明。

Rollback strategy: 如果下游 dashboard 迁移受阻，用户可临时设置 `output.tensorboard.legacy_accuracy_tags=true` 恢复旧 tag；代码层回滚只需恢复 writer 默认写入通用 tag。

## Open Questions

无。
