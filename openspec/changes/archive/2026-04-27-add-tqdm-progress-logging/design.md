## Context

训练入口目前由配置驱动，核心训练循环位于 `src/kd_sensing/engine/trainer.py`。训练结束后会保存 `train_log.json`、`training_outputs.npz`、TensorBoard 标量日志和训练曲线，但训练过程中没有统一的终端进度条，`train_log.json` 也只保存指标数组，缺少可直接阅读的 epoch 级进度摘要。

项目已在 `pyproject.toml` 中声明 `tqdm` 依赖，并在预处理代码中使用，因此训练流程可以复用现有依赖，不需要引入新的包。

## Goals / Non-Goals

**Goals:**

- 为训练流程提供 `tqdm` epoch/batch 进度条，显示当前训练进度和关键指标。
- 在运行目录日志中保存每个 epoch 的进度摘要，包括训练指标、验证指标、学习率和 epoch/batch 进度信息。
- 保持已有 `history` 指标数组、TensorBoard 标量写入、checkpoint、训练曲线和 dry-run 行为兼容。
- 提供配置开关，允许在 CI、重定向输出或非交互环境中关闭可视化进度条，但仍保存日志。

**Non-Goals:**

- 不改变训练损失、蒸馏逻辑、验证指标或模型构建语义。
- 不替换 TensorBoard，也不改变已有 event 文件字段。
- 不新增实时 Web UI、分布式训练进度聚合或跨进程日志系统。

## Decisions

- 使用 `tqdm.auto.tqdm` 包装 epoch 和训练 dataloader。
  - 原因：`tqdm.auto` 能在终端和 notebook 场景中选择合适显示方式，项目已经依赖 `tqdm`。
  - 备选：手写 `print` 进度或 Python `logging`。这些方案对长训练输出不够稳定，且难以提供 ETA 和刷新控制。
- 进度条显示采用 `set_postfix` 更新当前运行均值。
  - 原因：当前训练循环已经维护 `running_loss`、`running_task_loss`、`running_distill_loss` 和 `running_acc`，可直接复用，不改变指标计算。
  - 备选：每个 batch 写一行日志。该方案会显著增加日志体积，也容易影响终端可读性。
- `train_log.json` 保持原有 `history` 顶层指标数组，并新增结构化 `epoch_logs`。
  - 原因：保留现有消费者对 `history` 字段的兼容性，同时提供更易读的 epoch 级摘要。
  - 备选：改写 `train_log.json` 为全新 schema。该方案会破坏已有脚本或人工查看习惯。
- 增加输出配置项，例如 `output.progress.enabled`，默认启用。
  - 原因：用户默认获得进度反馈；测试或批处理环境可通过配置关闭显示，避免进度控制字符污染日志。
  - 备选：自动检测 `sys.stderr.isatty()` 后默认关闭。该方案在用户希望把进度写入作业日志时不够可控。

## Risks / Trade-offs

- `tqdm` 在非交互日志中可能产生控制字符 → 通过 `output.progress.enabled` 配置允许关闭显示，并在测试中覆盖禁用场景。
- 嵌套 epoch/batch 进度条可能输出较多内容 → 默认使用稳定的描述和 `leave` 策略，避免每个 batch 额外打印普通日志。
- 扩展 `train_log.json` 可能影响依赖完整 schema 的外部脚本 → 保持既有 `history` 指标数组不变，只追加新字段。
- 训练中断时可能无法写出最终完整日志 → 在当前训练流程的 `finally`/训练结束写入路径中集中关闭进度条和 writer，并在已完成 epoch 后更新 `epoch_logs`。

## Migration Plan

1. 在配置默认值中加入进度条开关，默认启用。
2. 在训练循环中引入 `tqdm.auto.tqdm`，包装 epoch 和训练 batch 迭代。
3. 使用 `set_postfix` 展示当前运行均值和学习率。
4. 在每个 epoch 完成验证和 TensorBoard 写入后追加 `epoch_logs` 记录。
5. 保持 `train_log.json` 中原有历史指标字段，并将 `epoch_logs` 一并写入。
6. 使用 `conda run -n kd_mm_beam` 执行 dry-run 或聚焦测试，验证日志字段和关闭进度条配置。

回滚策略：移除进度条包装和 `epoch_logs` 追加字段，保留原有 `history` 写入逻辑即可恢复旧行为。
