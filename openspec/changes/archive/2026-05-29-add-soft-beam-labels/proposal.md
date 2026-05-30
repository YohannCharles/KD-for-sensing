## Why

当前 beam selection 训练仍以 one-hot hard label 作为主监督。MMW Town10 这类 64-beam 任务在严格 split 下容易出现长尾类、少样本类甚至 train 未覆盖但 test 出现的 beam；只用 hard CE/focal 会把相邻 beam 的物理连续性丢掉，导致 unseen 或邻近 beam 的学习信号不足。

## What Changes

- 为 beam selection 训练增加一等 soft beam label 支持：source 侧优先使用每个 future frame 的 beam power/RSS 向量归一化成 oracle 分布，缺失时使用按 beam 角距离的 Gaussian smoothing 分布；target 快速适应侧只根据 hard beam label 和码本邻接关系生成 circular Gaussian soft label。
- 主 beam 监督 loss 支持 soft target distribution；hard `target_beam` 保留用于 top-k、DBA、checkpoint 选择、split 诊断和 viewer 对齐。
- 数据集 batch 新增 `target_beam_distribution` 与可选有效 mask，训练 supervised loss 在启用时优先消费 soft distribution。
- MMW/DeepSense6G 风格 source beam power 文件可直接产生 soft labels；target split 不读取 target-side power/RSS oracle，避免把评估期可用的 target power profile 泄漏进适应监督。
- 不移除 hard label 指标，不改变现有 top-k/DBA 口径。

## Capabilities

### New Capabilities

- `soft-beam-label-training`: 定义 beam soft label 的数据契约、生成策略、训练 loss 消费和 hard-label 指标保留要求。

### Modified Capabilities

- 无。该 change 作为新增训练能力接入现有 workflow，不删除既有 hard-label 训练路径。

## Impact

- 影响数据集输出：`DeepSense6GDataset`/`MMWDataset` 及相关 batch preparation。
- 影响训练：主 task loss、KD distiller 的 supervised loss、CRAF 辅助/反事实 beam loss；验证/评估 loss 和指标继续使用 hard `target_beam`。
- 影响配置：默认配置和 canonical recipe 需要暴露 soft-label 开关、source/target 生成策略与生成参数。
- 影响测试：需要覆盖 soft distribution 生成、soft focal/CE loss、训练/验证 fallback 和 MMW beam power oracle。
