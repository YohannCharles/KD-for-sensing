## Context

现有 H2R-JointCE 从冻结 CurrentControl checkpoint 校准 922 个 Router 参数。其 `full` 证据同时包含普通置信度统计与 prototype topology 统计，校准采用 40 epoch、三次主干视图和五项 Router 辅助量，因此当前结果不能区分 prototype 因果贡献，也不能判断复杂训练是否必要。

本轮继续复用现有 `PrototypeReliabilityRouter`、paired Joint panel、初始化/冻结契约和训练入口。筛选只改变 Router 证据屏蔽、loss 权重和 epoch，不复制模型或训练循环。

## Goals / Non-Goals

**Goals:**

- 在 Router 参数量不变的前提下比较普通置信度证据与 prototype-topology 证据。
- 比较 JointCE-only、JointCE+单一 topology monotonic 与当前完整 loss 栈。
- 检查 10 epoch 校准能否替代 40 epoch，并保留一个 40 epoch 完整复现锚点。
- 使用固定 GPU0--7、seed1、batch64、checkpoint、Joint panel 和 mask/evaluation cache。

**Non-Goals:**

- 不在本轮重构 paired view 为两次主干 forward；只有 Lite 证明损失可删后才值得改变训练 runtime。
- 不修改 canonical T2/S1，不补多 seed，不改变 corruption 算法和正式 outer split。
- 不把 seed1 inner 结果写入正式 claim。

## Decisions

### 1. 固定宽度 evidence profile

`router_variant_config.evidence_profile` 只允许 `full`、`generic_confidence` 和 `prototype_topology`，默认 `full` 保持已有 checkpoint/config 行为。profile 通过乘固定常量 mask 屏蔽证据，不改变 MLP 输入维度和参数量：

- `full`：保留全部 9 个 frame evidence 和全部窗口级 Router features。
- `generic_confidence`：保留 reliability、entropy、confidence、Top1--Top2 margin、logit norm、temporal cosine disagreement；屏蔽 circular coordinates 与 topology dispersion。
- `prototype_topology`：frame 只保留 prototype confidence、margin、circular coordinates 和 dispersion；窗口级只保留 prototype margin，其余位置置零。

选择固定 mask 而不是构建三个 Router 类，是为了参数公平、state-dict 兼容和最小实现。profile 必须写入模型 metadata、resolved config 与 launcher manifest。

### 2. JointCE 可直接训练 H2R 帧门控

H2R 的时间权重参与 pooled unimodal logits 和最终 fused logits，因此正权重的 JointCE 已能向 frame-health MLP 反向传播。配置解析不再强制 H2R 的 `frame_rank_weight>0`；非 H2R 候选仍不得启用 frame-rank。默认生成配置保持原值，历史行为不变。

### 3. 八卡矩阵只改变三类变量

固定映射如下：

| GPU | 候选 | profile | 监督/辅助 | epoch |
|---:|---|---|---|---:|
| 0 | Full-40 | full | beam-power full stack | 40 |
| 1 | Full-10 | full | beam-power full stack | 10 |
| 2 | Lite-10 | full | JointCE only | 10 |
| 3 | LiteMono-10 | full | JointCE + topology monotonic | 10 |
| 4 | Generic-10 | generic_confidence | JointCE only | 10 |
| 5 | GenericMono-10 | generic_confidence | JointCE + topology monotonic | 10 |
| 6 | Prototype-10 | prototype_topology | JointCE only | 10 |
| 7 | PrototypeMono-10 | prototype_topology | JointCE + topology monotonic | 10 |

Full 候选保持 quality=0.2、monotonic=0.2、frame-rank=0.2、anchor=0.01。Lite 候选使用 `label_topology`、JointCE weight=1.0，quality/frame-rank/anchor 为零，Mono 行仅额外启用 monotonic=0.2，因此不加载 future beam power。

### 4. 复用现有评估器

训练完成后，每个 checkpoint 使用相同 81-condition Joint cache 运行现有 joint-stress evaluator，报告 Uniform、candidate static prior、frozen Current、Dynamic 和 Oracle。晋级以绝对 ADBA、相对 Current/Uniform、相对自身 static prior、normalized gain 非劣和受损块降权率共同判断；不事后修改阈值。

## Risks / Trade-offs

- [profile 通过置零而非缩小网络，零输入维度仍带 LayerNorm 统计] → 三组使用完全相同归一化和参数量，差异仍只来自可用证据；结果按受控 feature ablation 解释。
- [Full-40 重复已有结果] → 它用于验证新 profile 默认路径和当前源码下的复现性，避免拿旧源码 checkpoint 与新候选混比。
- [本轮仍是三次主干 view] → 先判定 Lite 是否成立；若成立，后续单独 change 合并 main/control，避免同时改变损失和数据流。
- [seed1 偶然性] → 所有结果保持 development-only；只有用户确认最终候选后补 seed2--5。

## Migration Plan

1. 添加 profile 与配置测试，默认 `full` 数值兼容。
2. 添加八卡 launcher，dry-run 冻结 config/manifest identity。
3. 完成聚焦测试、OpenSpec strict validation 和单 batch smoke。
4. GPU0--7 并行训练，随后在同 GPU 映射上运行固定 Joint evaluation。
5. 任一失败只重跑对应候选；canonical 配置和旧输出不受影响。

## Open Questions

- Lite 胜出后，是否继续将 main/control 合并为两次 forward，由本轮结果决定，不在本变更预先实现。

