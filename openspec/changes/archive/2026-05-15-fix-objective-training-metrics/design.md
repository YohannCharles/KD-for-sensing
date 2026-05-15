## Context

`add-first-class-prediction-tasks` 已经把 `experiment.objective` 接入训练、验证、配置和 artifact metadata。现在发现两个实验解释问题：

1. objective-aware `multitask` 默认权重为 `beam=1.0, occlusion=1.0, position=0.01`，导致 position 分支几乎不影响共享表示，不能被解释为三任务等权多任务训练。
2. trainer 会把未启用或未计算的 auxiliary 指标写入历史数组和 TensorBoard，值为 `0.0`。这些曲线不是真实性能，却很容易被误读为某任务达到零误差或零 F1。

同时，项目原有“默认 early stopping 使用 DBA”的要求是针对 beam 预测任务建立的。objective-aware 单任务需要保留各自的主指标：occlusion 看 blocked F1，position 看 RMSE，否则 checkpoint 选择会重新偏向 beam。

## Goals / Non-Goals

**Goals:**

- 让 objective-aware `multitask` canonical 配置默认使用三任务等权 loss。
- 明确 objective-specific early stopping 默认值，并修正与旧 beam-only DBA 默认要求之间的语义冲突。
- 让 trainer、validator、TensorBoard、`train_log.json` 和 `training_outputs.npz` 区分真实计算的 metrics 与 inactive metrics。
- 保留用户显式覆盖 multitask 权重和 early stopping 指标的能力。
- 用回归测试覆盖配置解析、loss 权重、早停默认值、日志输出和 TensorBoard tag 写入。

**Non-Goals:**

- 不引入 GradNorm、PCGrad、uncertainty weighting 等动态多任务优化算法。
- 不改变 beam、occlusion 或 position target 的生成语义。
- 不要求历史已生成的 outputs 目录重写 TensorBoard event 或 metrics 文件。
- 不改变用户显式配置的非等权 multitask loss；只修正默认 canonical 行为。

## Decisions

### Decision 1: objective-aware multitask 默认三任务等权

`_objective_loss_config("multitask")` 和对应 virtual canonical 配置必须生成：

```yaml
loss:
  objective:
    weights:
      beam: 1.0
      occlusion: 1.0
      position: 1.0
```

用户仍可用 CLI override 或实体 YAML 显式设置其它权重。这样默认实验回答的是“同一 backbone 在三项任务等权监督下的行为”，而不是 beam/occlusion 优先的辅助训练。

Alternatives considered:

- 保持 `position=0.01` 避免 position 数值主导：这会继续让 position 分支弱到不可解释，违背多任务对照目的。
- 对三个 loss 做动态归一化：更复杂，需要额外算法设计和验证，不适合作为本次修复默认值。

### Decision 2: early stopping 默认值按 objective 解释

默认规则改为：

| objective | 默认 metric | mode |
| --- | --- | --- |
| `beam` | `val_adba` | `max` |
| `occlusion` | `val_occlusion_blocked_f1` | `max` |
| `position` | `val_position_rmse` | `min` |
| `multitask` | `val_multitask_loss` | `min` |

这里不是否定 DBA 默认，而是把 DBA 默认限定到 beam objective。配置加载时如果用户显式覆盖 early stopping metric 和 mode，继续尊重覆盖，但必须验证该 metric 在当前验证结果中真实可用。

Alternatives considered:

- 对所有 objective 继续使用 `val_adba`：会让 occlusion/position 单任务 checkpoint 按未优化的 beam logits 选择。
- 对 multitask 使用 ADBA：会偏向 beam，无法反映三任务等权训练目标。

### Decision 3: inactive metric 不写成真实零值

验证阶段只为实际具备 head、target 和 valid sample 的任务产出 auxiliary metrics。训练历史可以保留固定 key 以兼容旧消费者，但 inactive 值必须表示为缺失/不可用，而不是 `0.0`：

- `metrics.json`：未计算的 top-level auxiliary metric key 应省略，或在显式 availability 字段中标记为 inactive。
- `train_log.json`：epoch log 中未计算的 objective metric 使用 `null` 或省略，不能使用 `0.0`。
- `training_outputs.npz`：如需保留固定数组 key，inactive slot 使用 `NaN`。
- TensorBoard：只写入本 epoch 真实可用且 finite 的 scalar tag；不得为 inactive task 写入 `position/rmse=0`、`occlusion/accuracy=0` 等曲线。

Alternatives considered:

- 继续写 `0.0` 并靠文档解释：成本低，但实验图仍会误导。
- 删除所有 optional metric key：最干净，但可能破坏依赖固定数组 key 的旧分析脚本。因此设计保留训练输出数组 key 的兼容选项，但用 `NaN/null` 表示不可用。

### Decision 4: validator 提供 metric availability

validator 应输出足够信息让 trainer 判断哪些指标真实可用。可以采用以下任一实现，只要 artifact 行为满足规格：

- 在 `metrics` 中加入 `available_metrics` 列表；
- 或让缺失指标完全不出现，并让 trainer 用 key 是否存在判断；
- 或在 `metrics["objective"]`/`metrics["auxiliary"]` 中记录 enabled/computed 状态。

关键约束是 early stopping source validation 必须基于真实可用指标，而不能被 inactive 默认零值绕过。

## Risks / Trade-offs

- [Risk] position loss 数值尺度可能大于 beam/occlusion，等权后 multitask 训练初期可能更不稳定。→ Mitigation: 保持用户显式 override 能力，并在 README 说明如果 position 支配 loss 可进行权重 sweep。
- [Risk] `NaN/null` 会影响旧脚本。→ Mitigation: TensorBoard 不写 inactive tag；`training_outputs.npz` 保留 key；测试覆盖 JSON 和 NPZ 读取。
- [Risk] 修改 top-level early stopping 规格可能看起来与“默认 DBA”冲突。→ Mitigation: 明确 DBA 是 beam objective 默认，非 beam objective 使用各自主指标。
- [Risk] 历史 outputs 仍包含 `0.0` inactive 曲线。→ Mitigation: 文档说明变更只影响新训练，旧结果需要按 final_config 的 objective 解释。

## Migration Plan

1. 修改 objective-aware canonical loss defaults，让 `multitask` 默认三任务等权。
2. 校验 `configure_objective_defaults`、canonical config generator 和 final config runtime metadata，确保 beam/occlusion/position/multitask early stopping 默认值正确。
3. 修改 validator/trainer 的 optional metric 传递和记录逻辑，避免 inactive metrics 被填成 `0.0`。
4. 修改 TensorBoard writer，只写入 finite 且 active 的 optional scalar。
5. 更新 README 或实验说明，解释 objective-aware 指标和历史 inactive 零值的区别。
6. 增加测试并运行目标验证命令。

Rollback strategy: 如果等权 multitask 对现有实验不稳定，用户可显式覆盖 `loss.objective.weights.position=0.01` 恢复旧行为；代码层回滚只需恢复默认权重和 optional metric 填充值。

## Open Questions

- `metrics.json` 对 inactive metric 是完全省略，还是保留 `available_metrics`/`inactive_metrics` 元数据更利于下游分析；实现阶段可选择兼容性更好的形式。
