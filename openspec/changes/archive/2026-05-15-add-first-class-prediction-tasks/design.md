## Context

项目当前把 `experiment.task` 用作输入路由：`image`、`radar`、`gps`、`lidar`、`mmwave` 和 `fusion` 决定 dataset batch 准备、模型 forward 路径和 modality 输入。`add-occlusion-position-heads` 已经把遮挡标签、位置目标、CLS-token auxiliary heads、多任务辅助 loss 和辅助指标接入 fusion 训练，但 `occlusion` / `position` 仍是 beam 任务上的附属分量。

这对“多任务模态失衡”研究不够干净。单任务遮挡/位置实验需要间接关闭 beam focal loss，日志和 early stopping 仍以 beam 路径为中心，配置命名也无法表达“同一模态集合 × 不同预测任务”的实验矩阵。新的设计目标是把预测目标抽象出来，让 `beam`、`occlusion`、`position` 和 `multitask` 都成为可验证、可评估、可命名的一等任务。

## Goals / Non-Goals

**Goals:**

- 保留 `experiment.task` 作为输入路由，新增 `experiment.objective` 作为预测目标。
- 为 `beam`、`occlusion`、`position`、`multitask` 定义统一 objective contract：target 准备、output 选择、loss 计算、metrics、early stopping 和日志字段。
- 让 `occlusion` / `position` 单任务不再依赖 `loss.alpha: 0.0`，而是直接使用对应 head 和对应主 loss。
- 支持多任务模态失衡实验矩阵：同一 fusion 模态集合下切换 objective，并比较 single/strong/weak/all modality subsets。
- 保持现有 beam-only 配置兼容；未配置 `experiment.objective` 时默认 `beam`。

**Non-Goals:**

- 不把 `experiment.task` 重命名或改成 `occlusion` / `position`，避免破坏现有输入路由和训练入口。
- 不要求所有单模态模型首期支持 `occlusion` / `position` objective；首期以 CLS-token fusion 为主，其他模型可在校验中给出清晰错误。
- 不实现新的多任务优化算法，如 GradNorm、PCGrad、uncertainty weighting；本变更只提供一等任务基础设施和静态权重。
- 不改变遮挡标签和位置目标的生成语义；继续复用训练 split threshold、future GPS local XY 和 artifact 复用机制。

## Decisions

### Decision 1: 用 `experiment.objective` 而不是复用 `experiment.task`

配置采用：

```yaml
experiment:
  task: fusion
  objective: occlusion
```

`task` 继续选择输入和模型 forward 路由，`objective` 选择监督目标和评估语义。缺省 `objective` 为 `beam`，从而保持旧配置行为。

Rationale: 现有 `task=fusion` 已经被数据加载、batch 准备、forward kwargs 和 canonical config 大量使用。如果把 `task` 改成 `occlusion`，系统还需要另一个字段表达 fusion 输入，迁移成本和破坏面更大。

Alternatives considered:

- `experiment.task: occlusion` + `experiment.input_mode: fusion`：语义直观，但会影响所有读取 `experiment.task` 的旧代码。
- `loss.objective`：实现局部，但 objective 不只是 loss，还包括 targets、metrics、early stopping 和配置命名。

### Decision 2: 引入窄的 objective 调度层

新增 objective helper 或 registry，暴露少量稳定函数：

- `resolve_prediction_objective(cfg) -> beam|occlusion|position|multitask`
- `prepare_prediction_targets(batch, cfg, device) -> PredictionTargets`
- `compute_prediction_loss(model_output, targets, cfg, reference) -> LossBundle`
- `collect_prediction_metrics(outputs, targets, cfg, dataloader) -> dict`
- `primary_metric_for_objective(cfg) -> metric alias`

训练主循环不应散落大量 `if objective == ...` 分支，而是从该调度层获取 loss 和 diagnostics。现有 auxiliary loss helper 可以被迁移或包在 objective helper 内。

Rationale: 预测任务会横切 batch、trainer、validator、evaluator 和 logging。集中调度能让新任务扩展更可控，也减少 beam 逻辑与 occlusion/position 逻辑相互污染。

Alternatives considered:

- 在 trainer/validator 中直接分支：短期快，但会让每个流程重复处理 target/output/loss/metric。
- 每个 objective 新建训练脚本：会破坏项目统一入口和配置体系。

### Decision 3: `occlusion` / `position` primary objective 只优化对应主 loss

当 `experiment.objective: occlusion` 时，总 loss 默认只包含遮挡 BCE；beam logits 可以继续产生用于诊断，但 beam CE 不参与反传。当 `experiment.objective: position` 时，总 loss 默认只包含位置 MSE/SmoothL1；beam CE 不参与反传。`multitask` 才按权重组合 beam、occlusion 和 position。

Rationale: 一等任务实验必须能回答“这个模态集合对该目标本身有什么贡献”。如果继续隐式包含 beam loss，occlusion/position 单任务结果会混入 beam 优化目标。

Alternatives considered:

- 保留 beam loss 但权重可调：适合 auxiliary 研究，不适合一等单任务 baseline。
- 继续用 `loss.alpha: 0.0`：可跑但不可读，且 focal loss 的 `alpha` 本来是类别/损失参数，不应承担 task weight 语义。

### Decision 4: 模型输出使用同一 multi-head contract

CLS-token fusion 继续输出 beam logits，并在启用 head 时输出 `occlusion_logits` 和 `position`。Objective helper 根据 `experiment.objective` 选择需要的输出。对于 `occlusion` / `position` primary objective，配置校验必须要求模型具备对应 head；对于 `beam` objective，辅助 head 可关闭。

Rationale: 共享 backbone + 多 head 正好匹配多任务模态失衡研究。保留 beam logits 也能在单任务实验中记录诊断指标，但不把它作为主监督。

Alternatives considered:

- 为 occlusion/position 建独立模型类型：更显式，但会重复 fusion backbone，并削弱多任务共享表示对比。
- 修改 `ModelOutput.logits` 指向不同任务输出：会破坏现有 Top-K/DBA 假设，也让类型语义混乱。

### Decision 5: 配置矩阵按“模态 slug + objective”命名

新增推荐命名：

```text
configs/fusion/<slug>_<objective>_no_kd.yaml
```

例如：

- `configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml`
- `configs/fusion/image_radar_gps_lidar_mmwave_occlusion_no_kd.yaml`
- `configs/fusion/image_radar_gps_lidar_mmwave_position_no_kd.yaml`
- `configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml`

旧配置如 `configs/fusion/all_modalities_no_kd.yaml` 和 `configs/fusion/token_transformer_all_modalities_multitask_no_kd.yaml` 保留为兼容入口或文档 alias。

Rationale: 模态失衡实验需要横向比较不同模态组合和不同目标，文件名必须直接表达两个维度。

Alternatives considered:

- 只靠 CLI override 切换 objective：减少文件数量，但实验记录和复现性较差。
- 继续使用 `token_transformer_*` 前缀：模型类型信息太突出，弱化了实验矩阵的主维度。

## Risks / Trade-offs

- [Risk] objective 层与现有 auxiliary helper 重叠，短期有两套术语。Mitigation: 保留旧字段兼容，但 README 和 canonical 配置只推荐 `experiment.objective`；实现时让 auxiliary helper 逐步委托给 objective loss。
- [Risk] occlusion/position 单任务仍需要 beam label 或 beam power 文件来构建 dataset。Mitigation: 明确这是数据契约的一部分；occlusion 依赖 future beam power，position 依赖 future GPS target，batch 中不需要的 targets 不参与 loss。
- [Risk] early stopping 指标跨任务不可比较。Mitigation: 每个 objective 定义自己的默认主指标：beam 用 `val_adba`，occlusion 用 `val_occlusion_blocked_f1`，position 用 `val_position_rmse`，multitask 默认可配置。
- [Risk] 配置矩阵膨胀。Mitigation: 物理 YAML 只提供 recommended all/strong/weak 入口，其余通过 virtual canonical generator 解析。
- [Risk] 旧训练日志消费脚本假设 `train_task_loss` 等于 beam loss。Mitigation: 保留字段但把语义改为当前 objective 的 primary loss，并新增 objective-specific 字段用于消歧。

## Migration Plan

1. 增加 `experiment.objective` 解析和配置校验，默认值为 `beam`。
2. 引入 objective helper，先包装现有 beam loss 和辅助 occlusion/position loss。
3. 改造 trainer/validator/evaluator，使主 loss、metrics、early stopping 由 objective helper 驱动。
4. 扩展 CLS-token fusion 配置校验，要求 primary occlusion/position objective 必须启用对应 head 和 dataset target。
5. 增加 canonical/virtual objective 配置矩阵和 README 运行命令。
6. 增加 smoke/regression tests，验证旧 beam 配置不变、新 occlusion/position 单任务不再需要 `loss.alpha: 0.0`。

Rollback strategy: 如果新 objective 路径有问题，可以删除或忽略 `experiment.objective` 配置，旧配置会继续以默认 `beam` 路径运行。实现期间不删除现有 auxiliary 配置字段。

## Open Questions

- `multitask` 的默认 early stopping 应使用 `val_multitask_loss`、`val_adba`，还是显式要求用户配置；建议首期默认 `val_multitask_loss` 最小化，并允许覆盖。
- `position` 主 loss 默认使用 MSE 还是 SmoothL1；建议配置默认 MSE，保留 `loss.position.type: smooth_l1` 选项。
- 物理 YAML 是否只提供五模态 all objective 矩阵，还是同时提供 strong-only/weak-only；建议先提供五模态和 strong-only，其他通过 virtual config。
