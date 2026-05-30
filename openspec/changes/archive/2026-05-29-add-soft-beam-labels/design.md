## Context

现有训练流程以 `target_beam` hard label 作为主监督。`loss.beam_soft` 已存在，但它只是 CRAF extra loss 中的低权重正则，且基于 hard label 生成 Gaussian 分布，不能替代主 task loss，也不能利用 MMW 每帧已有的 64-beam power/RSS profile。

MMW prepared CSV 的 `future_beam*` 指向 beam power 文件，hard label 由 `argmax(power)` 得到。因此 source 训练 soft label 可以在 Dataset 中由同一文件归一化得到，并和 hard label 保持同源。target 快速适应阶段不能读取 target-side power/RSS oracle profile，只能由少量 hard beam label 和码本邻接关系生成 circular Gaussian soft label，避免把 target oracle 分布泄漏进适应监督。

## Goals / Non-Goals

**Goals:**

- 训练主 beam loss 能直接消费 `[B, H, C]` soft target distribution。
- Dataset 在启用 soft label 时返回 `target_beam_distribution`；source 域优先由 beam power/RSS 归一化得到，target 域只由 hard label 生成 circular Gaussian。
- source beam power 缺失或不可用时，能按 hard beam label 生成 circular Gaussian soft distribution。
- hard label 指标、split metadata、top-k/DBA 和 viewer 语义保持不变。
- no-KD、logits KD 和训练阶段 CRAF auxiliary/counterfactual 使用同一 soft-target 准备逻辑。
- 验证/评估 loss、top-k/DBA、checkpoint 选择和 viewer 语义继续使用 hard label。

**Non-Goals:**

- 不把 target beam power/RSS 作为 sensing input 特征或 target adaptation 监督信号。
- 不改变 top-k/DBA 指标为 soft-label 指标。
- 不声称 soft label 可以完全解决 train 中没有 exact beam 的泛化；它只提供邻近 beam 或 power profile 的连续监督。
- 不重写 split 生成策略；split label 覆盖问题仍需单独处理。

## Decisions

1. **Dataset 产生 soft distribution，batch/engine 消费统一字段。**

   `target_beam_distribution` 是数据契约，shape 为 `[H, C]`，DataLoader 后为 `[B, H, C]`。这样训练流程和 future extension 不需要重复读 beam power 文件。备选方案是在 engine 中根据 future path 现读 power 文件，但 engine 不应依赖数据集文件布局。

2. **主 supervised loss 优先使用 soft target，hard label 保留。**

   `prepare_soft_beam_targets()` 在训练 batch 中发现 `target_beam_distribution` 时返回 soft target；distiller 使用 soft target 计算训练 supervised loss。`target_beam` 仍传入 `prepare_labels()`，用于 validation/evaluation loss、metrics、ignore mask、DBA 和日志。

3. **source/target soft label source 分离。**

   `data.dataset.soft_beam_labels.source` 控制 source 域生成，默认 `power_or_gaussian`：如果 beam power/RSS 向量有效，按非负值归一化；如果缺失、维度错误、全零或非有限，按 hard label 生成 circular Gaussian 分布。`target_source` 固定为 `gaussian`，target 域强制 circular Gaussian，并且不读取 target-side power/RSS 文件。这样 source 侧可以利用物理 oracle profile，target 快速适应侧只使用少量 beam label 和码本邻接关系，避免 target oracle 分布泄漏。

4. **loss 模块兼容 hard/soft target。**

   `FocalLoss` 接收 long hard target 时保持现有行为；接收 float `[N, C]` soft target 时计算 soft focal loss。这样 `loss.type=focal_loss` 不需要替换为新类型，也能保持配置兼容。

5. **CRAF 相关 beam loss 跟随 soft target。**

   CRAF 的训练阶段 unimodal auxiliary 和 counterfactual CE 用于衡量 beam 预测质量，应在 soft target 存在时使用同一分布，避免主 loss 和辅助 loss 对齐到不同监督信号。验证/评估阶段不消费 soft target。

## Risks / Trade-offs

- [Risk] source power 向量尺度不一致会导致分布过尖或过平。→ Mitigation: 配置提供 `temperature` 和 `epsilon`，默认保守使用非负归一化；后续可按数据集调参。
- [Risk] soft label 会提高相邻 beam 概率，但不能让完全未见 beam 的特征判别凭空出现。→ Mitigation: 结果仍报告 hard top-k/DBA，并继续要求 split 覆盖和 strict eligibility。
- [Risk] 额外读取 beam power 会增加 Dataset IO。→ Mitigation: 复用已有 beam path，增加轻量 distribution cache；只缓存 64 维向量。
- [Risk] 老实验可复现性受默认值影响。→ Mitigation: 配置保留 `loss.soft_targets.enabled` 和 `data.dataset.soft_beam_labels.enabled` 开关，可显式关闭回到 hard label。
