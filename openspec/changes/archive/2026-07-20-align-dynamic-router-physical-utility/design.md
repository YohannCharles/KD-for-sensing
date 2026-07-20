## Context

上一轮 seed1 inner 筛选中，八个动态 Router 相对 train-fit static prior 均为正向，但没有候选同时通过材料性、置信区间和受损时间块降权 Gate。H2R 已能稳定识别并降权受损帧，但其 normalized beam-power gain 仍未超过 frozen Current Router。训练日志显示连续 fused expected utility 接近 `0.99`，同时 quality 与 monotonic loss 继续上升，说明该目标对最终 fused-logit argmax 决策已缺少有效梯度。

本变更只调整 paired Joint view 上的融合决策监督，不改变模型输入、corruption metadata 隔离、冻结 expert、固定 panel 或评估 Gate。所有候选仍属于 inner-only seed1 方法开发证据。

## Goals / Non-Goals

**Goals:**

- 将 Router 训练目标直接对齐到最终 fused logits 的硬标签或物理 beam-power 决策。
- 用 PATR 与 H2R 两种最小架构、四种互斥目标形成可归因的八卡筛选矩阵。
- 保持真实小量级线性 beam power 的 `float32` 数值边界，并记录目标身份和源码身份。
- 记录目标 loss、连续 fused utility、有效 margin 比率和既有质量诊断，定位训练是否真正改变最终决策。

**Non-Goals:**

- 不修改 canonical T2/S1/baseline recipe，不重训冻结 encoder、head 或 prototype bank。
- 不扩展 CoRe、Unified-HPR 或任意目标组合，不事后修改上一轮 Gate。
- 不把 future beam power、corruption 类型、严重度或状态矩阵输入模型 forward。
- 本轮不直接补 seed2--5，也不把 inner 结果写成正式论文 claim。

## Decisions

### 1. 用单一枚举选择互斥 fused-logit 目标

`dynamic_router.fused_decision_objective` 只允许 `expected_utility`、`joint_hard_ce`、`power_soft_ce` 和 `power_top1_margin`。继续复用 `fused_utility_weight` 作为所选目标的权重，避免再引入一个等价权重字段。未声明时默认 `expected_utility`，从而保持历史配置数值兼容。

相比允许多个布尔开关同时启用，枚举能保证每个候选只有一个决策变量，manifest 也能完整冻结实验身份。

### 2. 三个新目标直接作用于 Joint fused logits

`joint_hard_ce` 对 Joint fused logits 与真实 beam label 计算交叉熵。它不要求 power target，用于验证饱和问题是否仅由连续效用造成。

`power_soft_ce` 先在 `float32` 中验证非负有限 beam power，再按行和归一化为概率分布，并计算 target distribution 对 fused log-softmax 的交叉熵。选择线性功率归一化而非额外温度 softmax，是为了不引入新的搜索维度，并保留真实候选波束之间的相对功率。

`power_top1_margin` 选择每个样本最高功率 beam 为正类，从 fused logits 中选择当前得分最高的非正类作为 hard negative，并施加按归一化功率差缩放的 hinge margin。只比较一个 hard negative 可以直接约束 argmax，同时避免全波束两两排序的二次开销。零功率和无有效功率差样本不进入 active margin。

连续 `expected_utility` 保留为对照和所有目标共享的诊断量。所有 power target 均 detach，只允许梯度进入候选 Router。

### 3. 固定 PATR/H2R × 四目标的八任务矩阵

PATR 代表上一轮具有较稳定 static-prior 小增益的窗口级路线，H2R 代表能够实际降权受损时间块的分层路线。CoRe 与 Unified-HPR 没有提供独立、足以抵消复杂度的证据，因此不进入本轮。

八个候选共享 source checkpoint、240-entry Joint panel、seed1、batch64、40 epoch、optimizer、冻结边界与 GPU0--7 映射。除 H2R 必需的 frame-rank 分支外，各候选的 quality、monotonic、anchor 权重保持一致；只改变 fused 决策目标。

### 4. 保持训练期特权信息边界

hard CE 候选不加载 beam power。其余三个候选中，`expected_utility` 对照和两个 power 目标使用 future beam power，但该张量只传给 loss；模型 forward、Router features 和输出 schema 均不得包含它。launcher 必须在 resolved config 与 manifest 中记录 `requires_beam_power`、objective、checkpoint SHA、panel SHA 和 loss 源码 SHA。

### 5. 先完成夜间训练，再使用原冻结评估协议决策

今晚仅启动八个 40 epoch 校准任务并保存完整 checkpoint、日志和 manifest。训练完成后复用同一 81-condition Joint cache，比较 Uniform、train-fit static prior、frozen Current、Dynamic 和 Oracle。只有通过既有材料性和非劣 Gate 的候选才进入 seed2--5；不得因为新目标调整阈值。

## Risks / Trade-offs

- [hard CE 与主任务 beam CE 表面重复] → 它只作用于相同 availability 的 Joint corrupted view，主任务 CE 作用于原训练视图，二者反事实范围不同。
- [线性功率软分布可能过尖或过平] → 首轮不搜索温度，先验证无额外超参数的物理目标；若失败则终止该分支而非事后调参。
- [Top1 margin 忽略次优波束整体结构] → 保留 power soft CE 作为结构完整对照，并记录 active ratio 与 power gap。
- [40 epoch 可能过拟合 inner panel] → 固定 panel、冻结 expert、保存逐 epoch validation，最终仍以独立固定评估 cache 判定。
- [八个候选均失败] → 接受“动态可靠性不构成独立核心创新”的结论，不继续堆叠 Router 组件。

## Migration Plan

1. 为配置解析和 loss 增加互斥 objective，默认保持 `expected_utility`。
2. 添加聚焦单元测试，验证 power 数值、梯度、hard negative、互斥配置和旧配置兼容。
3. 添加独立 launcher，生成八个 resolved config 与不可变 manifest，先 dry-run 再启动 GPU0--7。
4. 训练结束后运行冻结 Joint evaluator；只有用户确认后才补多 seed 或修改 canonical recipe。
5. 回滚时删除未跟踪输出并继续使用默认 `expected_utility`，历史 checkpoint 和配置无需迁移。

## Open Questions

- 若 hard CE 与 power 目标都通过 Gate，优先选择不依赖训练期特权 power 的 hard CE；只有 power 目标提供稳定且显著额外收益时才保留后者。
- 若 PATR 与 H2R 同时通过，只有 H2R 的时间块降权证据与物理指标增益均更强时才接受其额外复杂度。
