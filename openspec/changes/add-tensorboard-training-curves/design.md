## Context

当前训练入口由配置驱动，`train()` 会为每次运行创建 `outputs/<run_name>/`，并在训练结束后写入 `final_config.yaml`、checkpoint、`train_log.json`、`training_outputs.npz` 和静态训练曲线。训练循环已经在每个 epoch 汇总了 loss、accuracy 和 learning rate，因此 TensorBoard 所需的标量数据已经存在，主要缺口是 event 文件写入、配置项、依赖声明和文档说明。

## Goals / Non-Goals

**Goals:**

- 在训练过程中按 epoch 写入 TensorBoard 标量日志，支持实时查看训练曲线。
- 记录与现有 `history` 一致的核心曲线，避免引入新的指标语义。
- 通过配置控制 TensorBoard 是否启用以及日志子目录名称。
- 保留现有训练输出文件和静态曲线行为。
- 覆盖 dry-run 训练路径，确保最小训练流程也会生成 TensorBoard 日志。

**Non-Goals:**

- 不改变模型、loss、KD、optimizer、scheduler 或验证指标的计算方式。
- 不引入 TensorBoard 图结构、embedding、图片或特征分布记录。
- 不重构训练循环或输出目录结构。
- 不为评估入口新增 TensorBoard 记录。

## Decisions

- 使用 `torch.utils.tensorboard.SummaryWriter` 写 event 文件。该 API 与 PyTorch 训练流程匹配，避免引入额外日志框架；同时在依赖中声明 `tensorboard`，确保运行时可用。
- 在 `output` 配置下增加嵌套配置，例如 `output.tensorboard.enabled` 和 `output.tensorboard.log_dir`。该位置与现有 `output.dir`、`output.run_name` 语义一致，也能通过当前点号覆盖机制直接关闭或改名。
- 默认启用 TensorBoard，并将 event 文件写入 `run_dir/<log_dir>/`，默认 `log_dir` 为 `tensorboard`。这样每次实验的所有产物仍集中在同一个 run 目录，用户也可以用 `tensorboard --logdir outputs` 比较多个 run。
- 在每个 epoch 完成验证并更新 `history` 后写入标量，使用 `epoch + 1` 作为 global step。这样 TensorBoard 中的 step 与训练日志中的第几轮保持一致，并包含验证指标。
- 使用少量 helper 函数封装 writer 创建、标量写入和关闭逻辑，训练主循环只保留调用点。writer 必须在异常或 early stopping 路径下关闭，避免 event 文件未 flush。
- 标量 tag 使用稳定分组：`loss/train`、`loss/train_task`、`loss/train_distill`、`loss/val`、`accuracy/train`、`accuracy/val`、`learning_rate/main`。这些 tag 覆盖当前训练曲线，不改变历史文件 key。

## Risks / Trade-offs

- [依赖增加] 安装包需要额外 `tensorboard` 依赖 → 在 `pyproject.toml` 中显式声明，并保持使用 PyTorch 官方 writer 接口。
- [训练异常时 event 未写完] 异常或早停可能导致 writer 未 flush → 使用 `try/finally` 或等价结构关闭 writer。
- [输出文件增加] 每次训练会多生成 event 文件 → 提供 `output.tensorboard.enabled=false` 配置关闭。
- [tag 命名兼容性] 后续如果新增指标，tag 可能扩展 → 本次只记录现有 `history` 指标，避免扩大行为范围。

## Migration Plan

- 新增默认配置后，现有配置文件无需修改即可默认写入 TensorBoard 日志。
- 用户可通过 `output.tensorboard.enabled=false` 关闭日志写入，或通过 `output.tensorboard.log_dir=<name>` 修改子目录。
- 回滚时移除 writer 调用、配置项、依赖和 README 说明即可；已有 event 文件只是训练产物，不影响模型权重或旧日志读取。

## Open Questions

无。
