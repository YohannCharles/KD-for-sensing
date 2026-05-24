## Context

训练入口目前在 `_train_inner` 启动时构建一次 train/test DataLoader，train DataLoader 通过 `shuffle=True` 每个 epoch 遍历完整 train dataset。已有 `data.dataset.portion` 可以缩小 dataset 本身，但它会改变 dataset 构建、split 元数据和可能的归一化统计语义；用户想要的是保留原 train CSV 的同时，让每个 epoch 只跑一部分样本以缩短调试反馈。

训练输出已经记录 split 样本数、DataLoader 参数、throughput 元数据、epoch 日志和 final/resolved config。本 change 应沿用这些记录路径，并保持默认完整训练行为不变。

## Goals / Non-Goals

**Goals:**

- 增加显式配置，支持每个 train epoch 使用固定数量或固定比例的 train 样本。
- 默认禁用，现有配置、CLI 覆盖、checkpoint、验证和评估语义保持兼容。
- 子采样结果可复现；resume 后相同 epoch 必须得到相同样本集合。
- 每个 epoch 可轮换抽样，避免长期只学习同一个固定小切片。
- 在运行产物中记录完整 train split 样本数、有效子采样样本数、抽样 seed、策略和 epoch 轮换状态。

**Non-Goals:**

- 不替代 `data.dataset.portion`；后者仍用于构建更小 dataset 或 smoke test。
- 不改写 train/test CSV，不新增 split 文件，不改变 dataset `__getitem__` 返回字段。
- 不让验证/test split 跟随 train 子采样缩小。
- 不保证子采样训练指标可与完整训练直接横向比较；它主要用于快速迭代、调参和排障。
- 不在运行日志中保存每个 epoch 的完整样本 index 列表，避免日志膨胀；记录可复现所需的策略、seed 和计数。

## Decisions

1. 子采样放在 DataLoader sampler 层，而不是裁剪 dataset 或生成临时 CSV。

   `build_dataset` 继续构建完整 train dataset，保证 split metadata、模态解析、cache key 和 scaler/normalizer 输入语义不被隐藏改变。`build_dataloader` 在 train split 且配置启用时注入自定义 sampler；test split 继续使用现有 DataLoader 参数。替代方案是每个 epoch 重建 `Subset` 或写临时 CSV，但这会增加 worker 生命周期管理复杂度，也更容易污染运行元数据。

2. 新配置放在 `training.epoch_subsampling`。

   建议配置形态：

   ```yaml
   training:
     epoch_subsampling:
       enabled: false
       fraction: null
       num_samples: null
       seed: null
       rotate_each_epoch: true
       shuffle: true
   ```

   `fraction` 和 `num_samples` 二选一；`seed` 为空时使用 `experiment.seed`。`fraction` 必须满足 `0 < fraction <= 1`，`num_samples` 必须为正整数。有效样本数大于等于完整 train dataset 长度时退化为完整训练并记录该退化结果。放在 `training` 下的原因是它改变 epoch 训练计划，而不是 dataset 内容或 DataLoader worker 参数。

3. 使用可设 epoch 的无放回 sampler。

   新增类似 `EpochSubsampleSampler` 的小型 sampler，持有 dataset 长度、有效样本数、seed、`rotate_each_epoch` 和 `shuffle`。每次 `__iter__` 根据当前 epoch 和 seed 生成无放回 index 序列；trainer 在 epoch 开始前调用 `set_epoch(epoch)`。当 `rotate_each_epoch=false` 时始终使用 epoch 0 的选择，方便固定小样本调试。DataLoader 启用 sampler 时不再传 `shuffle=True`，由 sampler 负责顺序随机化。

4. resume 依赖绝对 epoch 编号保证可复现。

   checkpoint 恢复后训练循环已经从恢复的 `state.start_epoch` 继续。sampler 以绝对 epoch 编号生成子集，因此恢复运行中第 N 个 epoch 的样本集合与未中断运行一致，不需要在 checkpoint 中额外保存 sampler 状态。

5. 运行元数据记录策略而不是全量 index。

   `dataloaders_run_metadata` 和 `throughput_run_metadata` 增加 train 子采样字段，例如完整 train 样本数、有效 epoch 样本数、fraction/num_samples、seed、rotate_each_epoch、shuffle、sampler 版本和是否退化为 full epoch。`epoch_log` 增加当前 epoch 的有效 train 样本数和 sampler epoch。这样可以解释 loss 曲线和 samples/s，也不会把日志变成 index dump。

6. 归一化和 cache 语义默认仍基于完整 train dataset。

   GPS/LiDAR/mmWave/CSI 等 train-fitted artifact 继续按现有 dataset 构建路径处理。若用户希望连这些统计和 dataset 初始化也缩小，应继续使用现有 `data.dataset.portion` 或预先准备小 CSV。该取舍让 epoch 子采样保持“只缩短训练 step”的清晰语义。

## Risks / Trade-offs

- [Risk] 子采样导致训练损失和 early stopping 更噪声化。→ 在文档和运行元数据中明确标记子采样运行；验证仍使用完整 split，用户可先调参后再关闭子采样做正式训练。
- [Risk] 类别分布在小样本子集里不稳定。→ 首版采用无放回随机抽样并支持按 epoch 轮换；分层抽样作为后续扩展，不放入本 change。
- [Risk] 用户误以为它会减少 dataset 初始化、normalizer 拟合或 cache 预热时间。→ 文档写清它主要减少 epoch training step；需要缩小 dataset 构建时使用 `data.dataset.portion`。
- [Risk] DataLoader 同时传 `sampler` 和 `shuffle` 会报错。→ 构建 kwargs 时在启用 sampler 的 train split 显式移除/覆盖 `shuffle`。
- [Risk] 日志不保存具体 index 会降低事后审计精度。→ 记录 seed、sampler 版本、epoch 和有效样本数；若未来需要强审计，可增加可选 index manifest。

## Migration Plan

1. 增加默认配置字段，默认 `enabled=false`。
2. 实现 sampler 与配置解析校验，并接入 train DataLoader 构建。
3. 在训练循环 epoch 开始前设置 sampler epoch，并把当前有效样本数写入 epoch 日志。
4. 扩展运行元数据和 throughput 元数据，记录 train 子采样策略。
5. 更新 `docs/training_throughput.md` 或 README 相关示例，给出快速调试命令。
6. 增加单元测试和短训练 smoke test，运行 OpenSpec 校验。

## Open Questions

- 是否需要第一版支持分层抽样来稳定 beam class 分布。当前建议先不做，等真实小样本训练显示类别偏斜成为主要问题后再单独设计。
- 是否要提供可选 index manifest。当前方案只记录可复现参数，避免默认产物过大。
