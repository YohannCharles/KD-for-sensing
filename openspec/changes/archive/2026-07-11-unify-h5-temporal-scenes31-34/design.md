## Context

H5/P1 launcher 从不同方法的历史基配置生成训练配置。U-Mask 基配置已包含 Scene31-34 分层划分，而 AMBER 和 RMBP-MM 基配置仍是 Scene31-only；当前 overlay 只同步窗口、batch 和 temporal missing 参数，没有覆盖数据划分。

## Goals / Non-Goals

**Goals:**

- 由 launcher 在最终 merge 阶段为所有方法写入同一 Scene31-34 split contract。
- 让基线保留各自模型、优化器和 batch 配置，同时统一样本集合。
- 用 dry-run 测试阻止 Scene31-only 基配置再次泄漏到生成配置。

**Non-Goals:**

- 不修改本地 dataset 文件或历史 checkpoint。
- 不恢复已退役的 temporal-router 公共入口。
- 不声称新训练结果在完成评估前优于现有方法。

## Decisions

- 在 launcher 的最终公共 overlay 中写入 scenes、train/validation/test scenes、split protocol、strategy、seed、source splits 和 fractions。该位置优先级最高，可覆盖任意方法基配置，同时不复制完整 S1 YAML。
- 保留每个方法已有预处理字段和 dataloader 参数；统一的是数据与 split contract，不强制相同 batch size。
- 新训练写入独立 runtime root，避免覆盖 Scene31-only checkpoint，并允许失败后按现有 `--auto-resume` 机制续跑。

## Risks / Trade-offs

- [Scene31-34 训练显著延长基线运行时间] → 两方法并行运行并保留 checkpoint/日志。
- [历史基配置携带 Scene31 runtime metadata] → launcher 最终 overlay 只以 canonical `data.dataset` 字段为准，测试直接断言生成 YAML。
- [显存被其他任务动态占用] → 使用独立 GPU、较保守 batch，并在启动后检查 OOM 和进程存活。
