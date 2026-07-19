## Context

当前 `UMaskBeamJEPA` 先将 `[B,5,4,D]` 表征按 mask 池化为 `[B,4,D]`，再由窗口级 supervised Router 根据单模态 logits 的置信度、熵、margin 和 norm 输出四个权重。联合压力 trace 已证明该 Router 的绝对权重能反映长期模态强弱，但权重变化与当前样本的单模态效用变化相关性接近零，且不及同 checkpoint 的静态 Clean prior。

本轮是 inner-only seed1 方法筛选。八个任务必须共享成熟 CurrentControl checkpoint、MMW 15 域训练数据、固定 5×4 几何、训练预算和 Joint panel；只有 Router 方法族和监督来源不同。现有 canonical T2/S1、严格 resume、历史输出及 outer evidence 均不得被改写。

## Goals / Non-Goals

**Goals:**

- 用一套共享组件表达 PATR、H2R、CoRe 和 Unified-HPR，而不是复制四个模型或训练循环。
- 在任何时间池化前保留逐帧 prototype 质量证据，并让 H2R 的帧级门控能获得真实梯度。
- 将长期模态能力与样本级退化解耦为 train-fit global prior 和有界 dynamic residual。
- 用相同 availability 的 drop-control/joint-corrupt 反事实对训练动态权重，不把 corruption state、类型或严重度输入模型或 loss。
- 通过严格初始化、冻结、manifest 和固定 8 卡映射使候选可复算、可审计。

**Non-Goals:**

- 不把本轮 seed1 开发结果直接冻结为 canonical T2，不写正式 outer claim，也不补 seed2--5。
- 不重新训练 encoder、prototype bank、现有 unimodal head 或 Current Router；候选只校准新增 Router 模块。
- 不新增新传感器、第三方依赖、旧入口或独立训练器。
- 不把 corruption metadata、future beam power 或 GPS scaler送入模型 forward。

## Decisions

### 1. 单一 `router_variant` 枚举映射四个方法族

`model.primary.router_variant` 只允许 `current`、`patr`、`h2r`、`core`、`unified_hpr`。`current` 不实例化新模块，保持 state-dict key 和 forward 数值兼容；其余值实例化一个共享 `PrototypeReliabilityRouter`，内部按固定映射启用 temporal evidence、hierarchical gate 和 consensus evidence。拒绝任意布尔组合，避免无意义配置爆炸。

候选保持现有输出键。对于 H2R，`unimodal_logits` 必须是帧健康度池化后的四模态 logits，使 `fused_logits = Σ_m w_m z_m` 仍可由评估器重构；另输出 `[B,5,4]` 的 `router_temporal_weights` 用于机制图。

### 2. PATR 使用逐帧 prototype 统计和先验锚定残差

模型在冻结的 active head 上计算 `[B,5,4,64]` cell logits。每个 cell 的无标签质量证据包括归一化熵、置信度、Top1--Top2 margin、prototype circular/linear dispersion 和相对模态均值的特征余弦偏差。PATR 对这些量计算 masked mean/std、时间分布分歧和有效帧比例，然后与既有窗口可靠性特征拼接。

每个候选持有一个可训练的全局 `prior_logits[4]`，它仅在训练集校准阶段被优化。共享 residual MLP 不接收 modality one-hot，末层零初始化并输出有界残差：

```text
gate_logits = prior_logits + residual_scale * tanh(residual(features))
weights = masked_softmax(gate_logits, available)
```

因此模态身份只能进入静态 prior，动态分支必须根据本样本质量证据修正它。选择联合训练 prior 而非先扫描开发 trace，是为了保证 prior 只由训练数据拟合，并可在同 checkpoint 中直接作为静态反事实。

### 3. H2R 在池化前估计时间块健康度

H2R 的共享 cell-health MLP 读取逐帧无标签质量证据，末层零初始化。它在每个模态的有效时间块上做 masked softmax，得到 `[B,5,4]` 权重，再据此池化 latent sequence。无效 cell 权重严格为零；初始化时所有有效帧等权，因此 warm-start 行为稳定。

H2R 不是 20-cell flat softmax：帧级权重只在同一模态内部归一化，模态级 prior/residual 再在可用模态之间归一化。最终 cell 贡献是二者乘积，分别回答“这一帧是否健康”和“该模态当前是否可靠”。

### 4. CoRe 使用 leave-one-out prototype 共识

CoRe 对每个可用模态构造其余可用模态的平均 prototype 分布，并计算 JSD、按 active topology 解释的 Top1 距离和 Top-k overlap。只有一个模态可用时共识证据置零并回退到 masked prior。共识用于发现高置信但与其他传感器矛盾的模态，不把任何传感器指定为固定教师。

Unified-HPR 同时启用 H2R 帧级门控、PATR 时序证据和 CoRe 共识；模块仍由同一实现与同一 loss 驱动。

### 5. 两种监督共享连续 utility 接口

`label_topology` 将真实 beam label 按 active BPA topology 和 sigma 转为 Gaussian utility，按行最大值归一化后与 `softmax(logits/T)` 求期望。`beam_power` 将完整 future beam-power 向量按样本最大值归一化后计算相同期望。所有 expert logits 和 utility target 均 detach，梯度只进入 prior、residual、cell-health 等候选 Router 参数。

MMW beam-power 文件存储约 `1e-12--1e-8` 的线性功率。`beam_power` 分支必须在 AMP cast 之前将 logits 与 power target 提升到 `float32`，再完成逐样本最大值归一化和期望效用计算；不得先转 `float16`，也不得把线性功率重复转换为 dB。该数值策略作为筛选 provenance 记录，防止半精度下溢使不同 loss 分支处于不同物理尺度。

统一 loss 包含：主视图 beam CE、四模态 macro quality regression、融合 expected-utility loss和配对 residual monotonic loss。配对 loss 直接从 control/joint 单模态 utility 差确定 active 模态；近 tie 不产生单调约束，因此不读取 state matrix 或 affected-modality 标签。beam-power 缺失或 topology descriptor 不一致时 fail closed。

### 6. 配对 Joint panel 与 forward 无泄漏

保留原 600-entry missing 主调度作为主 forward。另生成内容寻址的 240-entry配对 panel：Joint20/40/60/80 各 60 条，每档 S1/S2/S3 各 20 条；Drop/Corrupt cell 在 panel 级对 cell、模态和帧精确平衡。每个训练样本按确定性全局 row index 选择一条 panel condition。

drop-control 与 joint view 从同一 preserved temporal superset 构造并拥有完全相同的 `[B,5,4]` availability：control 对 Corrupt cell 保持原值，joint 仅在这些 cell 上施加相应传感器 corruption。state、severity 和 condition id只写生成审计；统一 route API只接收 detached latent/cell logits/mask，loss只接收路由输出、label及可选 power。

### 7. 显式 initialization 与 calibration freeze

新增 `training.initialization_checkpoint`，与 `training.resume` 互斥。它在 optimizer 构建前验证 source path、SHA256、checkpoint role/schema和 key allowlist，只加载既有 expert/Current Router 权重；新增 Router key必须落入精确 missing-prefix allowlist。optimizer、scheduler、epoch、RNG、sampler和 extension state全部重新初始化，load report写入运行 provenance。

`router_calibration_only=true` 冻结 encoder、projection、reliability head、classifier、prototype bank、temporal pooling和 Current Router，并在每次 `model.train()` 后强制这些模块保持 eval。optimizer 必须 `require_all_matched=true` 且只包含新 Router 参数。

### 8. 固定 8 卡 seed1 筛选和晋级 Gate

GPU0--7 依次运行 PATR-label、PATR-power、H2R-label、H2R-power、CoRe-label、CoRe-power、Unified-label、Unified-power；共同使用 seed1、batch64、10 epoch、AdamW、`lr=1e-3`、相同 source checkpoint 和 panel checksum。launcher 先生成 resolved configs/manifest并做 checkpoint/GPU/preflight，再一卡一任务启动。

候选必须在 Joint40/60/80 上同时报告 Uniform、candidate train-fit prior、frozen Current Router、Dynamic 和 Oracle。晋级要求 Dynamic 相对 train-fit prior 的 ADBA 与 normalized gain 均为正且通过预注册 paired interval，同时 Clean/Drop-only 不超过非劣界；否则该方法只能作为负结果。

## Risks / Trade-offs

- [逐帧 head 和 paired view 增加训练开销] → 只运行10 epoch Router校准，冻结专家并复用同一共享实现；启动前用batch64 smoke测显存。
- [从成熟 checkpoint 校准不等于端到端训练] → 本轮只用于因果筛选；候选通过后再冻结40 epoch端到端配方并补5 seeds。
- [跨模态共识可能压制唯一正确模态] → prior提供回退，只有一个可用模态时禁用共识，并单列 minority-correct 分析。
- [beam-power 属于训练期特权信息] → 与不使用 power 的 label-topology 版本严格成对比较，论文明确披露。
- [线性 beam-power 在 AMP 下可能下溢] → 归一化与效用计算固定使用 `float32`，以真实小功率回归测试和源码 SHA 锁定数值策略。
- [八个候选都不超过静态 prior] → 不继续调小超参数；结论是当前数据不足以支持动态可靠性主张，回退为静态可靠性融合。

## Migration Plan

1. 保持 `router_variant=current` 默认并通过严格兼容测试。
2. 实现候选组件、utility、Joint panel和 initialization/freeze 契约，完成 focused smoke。
3. 生成8个 ignored resolved config和manifest，启动 seed1。
4. 训练完成后在同一 fixed-mask协议下运行机制评估；用户确认前不修改 canonical recipe。
5. 任一实现或训练失败可删除对应 ignored run 并回退到 current path，源码默认行为不受影响。

## Open Questions

- 若 label-topology 与 beam-power 都通过 Gate，正式主线优先选择不使用特权 power 的 label-topology；仅当 power 版本具有明确且稳定的实际收益时再保留后者。
- Unified-HPR 只有在显著超过更简单方法族时才晋级，否则按最小充分架构原则选择 PATR、H2R 或 CoRe。
