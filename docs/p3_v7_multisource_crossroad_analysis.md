# p3_v7_multisource_crossroad 分析笔记

本文汇总当前 `p3_v7_multisource_crossroad_seed0_*` 一组实验的模型结构、损失、训练方式、实验矩阵和初步结论，便于和其他模型或其他 AI 继续讨论。

> Retired note: P3/V7/V8/HiST-Beam 源码、配置和 shell wrapper 已从当前支持面删除。本文只作为历史实验分析保留，文中的运行命令、源码路径和 config 路径不再表示当前可运行入口。

涉及的主要结果目录：

- [`outputs/p3_v7_multisource_crossroad_seed0_iofix3_1x3090`](../outputs/p3_v7_multisource_crossroad_seed0_iofix3_1x3090)
- [`outputs/p3_v7_multisource_crossroad_seed0_v7_residual_free_probe_1x3090`](../outputs/p3_v7_multisource_crossroad_seed0_v7_residual_free_probe_1x3090)
- [`outputs/p3_v7_multisource_crossroad_seed0_v7_residual_1e4_probe_g1_1x3090`](../outputs/p3_v7_multisource_crossroad_seed0_v7_residual_1e4_probe_g1_1x3090)
- [`outputs/p3_v7_multisource_crossroad_seed0_v7_residual_1e3_probe_g2_1x3090`](../outputs/p3_v7_multisource_crossroad_seed0_v7_residual_1e3_probe_g2_1x3090)
- [`outputs/p3_v7_multisource_crossroad_seed0_v7_target_balance_probe_1x3090`](../outputs/p3_v7_multisource_crossroad_seed0_v7_target_balance_probe_1x3090)
- [`outputs/mmw_fixed_source_skybridge_long`](../outputs/mmw_fixed_source_skybridge_long)
- [`outputs/p3_v8_a2a5_single_target_seed0_4x3090`](../outputs/p3_v8_a2a5_single_target_seed0_4x3090)

## 1. 问题背景

目标是评估 MMW 场景下的 LOSO 跨场景 beam prediction。当前关注点是：

- target scene: `Town10_crossroad_seed24`
- source scenes: `Town10_skybridge_seed24`, `Town10_Hroad_seed42`, `Town10_curvyroad_seed42`
- label budget: `10`
- seed: `0`
- modalities: `image + gps + lidar`
- excluded sensitive fields: `mmwave, csi, channel, path, beam_power`

这组实验的核心问题不是“能不能跑通”，而是“多源 v7 是否真的比单源 baseline 更好”。

## 2. 当前模型架构

当前主线模型是 `HistBeamFusionNet` 的 `v7_shared_physical_private_residual` 变体。

### 2.1 主干结构

```text
image/gps/lidar
   -> modality encoders
   -> feature_projections
   -> transformer fusion
   -> fused representation
      -> shared_branch
      -> private_branch
      -> coarse_head / fine_head
      -> v7 heads
```

### 2.2 v7 输出

v7 的关键输出来自：

```python
logits_final = logits_shared + alpha * delta_logits_private
alpha = sigmoid(residual_gate(adapter_rep))
pred_beamspace_power = softmax(physical_beamspace_head(shared))
```

也就是说：

- `logits_shared`：共享分支的 beam logits
- `delta_logits_private`：private residual 修正项
- `alpha`：门控系数
- `logits_final`：最终 beam 预测

当前适配阶段主要只训练：

- `private_adapter`
- `private_residual_head`
- `residual_gate`

shared 主干和 shared beam head 都被冻结。

## 3. 损失函数

### 3.1 source 训练的 v7 loss

当前 v7 loss 的主项是：

- `v7_shared_ce`
- `v7_final_ce`
- `v7_bsp_kl`
- `v7_phys_kl`
- `v7_res_l2`
- `v7_gate_l1`
- `v7_diff`

默认权重来自 `configs/hist_beam/v7_shared_physical_private_residual.yaml`：

- `v7_shared_ce = 1.0`
- `v7_final_ce = 1.0`
- `v7_bsp_kl = 1.0`
- `v7_phys_kl = 1.0`
- `v7_res_l2 = 0.01`
- `v7_gate_l1 = 0.001`
- `v7_diff = 0.01`

另外，`shared_warmup_epochs=1`，warmup 阶段只保留 shared 侧 loss，final/residual/gate 不参与。

### 3.2 target adaptation loss

target adaptation 阶段没有 target physical oracle，因此：

- `bsp_kl = 0`
- `phys_kl = 0`
- 主要优化 `final_ce + v7_res_l2 + v7_gate_l1`

这意味着 target adaptation 的纠偏能力主要依赖 residual 分支，而不是 physical supervision。

### 3.3 target class balance

配置里虽然有 `hist_beam.class_balance`，但当前 quick validation / adaptation 路径里：

- `target_adaptation=false`
- 所以 target class balance 默认没有真正起作用

## 4. 训练方式

### 4.1 运行结构

当前 LOSO 运行由 `kd-sensing-hist-beam-loso` 驱动，阶段顺序是：

```text
source_train
source_only_target_test_eval
target_adaptation
adapted_target_test_eval
summary
```

### 4.2 训练节奏

当前 quick validation 中：

- source_train: 5 epochs
- target_adaptation: 5 epochs
- label budget: 10

target adaptation 的 few-shot 样本是 `beam_frequency_stratified_few_shot`，但样本仍然非常少，且每类分布很稀。

### 4.3 训练规模

v7 probe 的 trainable 参数只有约 `3.2e-3` 量级的 ratio：

- `trainable_params = 38627`
- `trainable_ratio = 0.0032178`

这很关键。它解释了为什么 residual 有时能移动方向，但很难把 top1 拉回来。

## 5. 实验结构

### 5.1 主对照

1. `iofix3` 原始 multisource v7
2. `v7_target_balance`
3. `v7_residual_free`
4. `v6_full_finetune`

### 5.2 4 组并行 probe

这 4 组是为了快速验证 residual 正则强度：

- `g0`: `v7_res_l2 = 0`
- `g1`: `v7_res_l2 = 1e-4`
- `g2`: `v7_res_l2 = 1e-3`
- `g3`: 当前这次跑出来的结果与 `g1` 基本重复，不是独立 target balance 对照

## 6. 结果汇总

### 6.1 单源 fixed-source 基线

来自 `outputs/mmw_fixed_source_skybridge_long` 的 target-only evaluation：

| 方案 | Top-1 | Top-3 | Top-5 | NRP | Beam loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v0_flat` | 0.0998 | 0.3047 | 0.5236 | 0.3887 | 5.2100 |
| `v3_decoupled` | 0.0963 | 0.2242 | 0.2680 | 0.3791 | 5.4168 |

这两条单源 baseline 反而比当前 multisource v7 更稳。

### 6.2 multisource v7

| 方案 | Top-1 | Top-3 | Top-5 | NRP | Beam loss | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `iofix3` 原始 v7 | 0.0070 | 0.0152 | 0.0420 | 0.2930 | 6.3283 | 严重塌缩 |
| `target_balance` | 0.0175 | 0.0245 | 0.0444 | 0.3044 | 6.2026 | 轻微改善 |
| `residual_free` | 0.0309 | 0.1477 | 0.3327 | 0.3265 | 6.0486 | Top-3/5 有明显提升 |
| `v7_res_l2=1e-4` | 0.0660 | 0.2172 | 0.3170 | 0.3418 | 5.9750 | 目前更好 |
| `v7_res_l2=1e-3` | 0.0753 | 0.2218 | 0.3421 | 0.3520 | 5.8789 | 当前最佳 |
| `v6_full_finetune` | 0.0000 | 0.0123 | 0.0344 | 0.2828 | 6.4868 | 失败更彻底 |

### 6.3 v8 target-prior A2-A5 诊断

这组来自 `outputs/p3_v8_a2a5_single_target_seed0_4x3090`，设置为：

- target: `Town10_crossroad_seed24`
- source: `Town10_skybridge_seed24`
- budget: `10`
- seed: `0`
- source_train: 20 epochs
- target_adaptation: 1 epoch
- variant: `v8_target_prior_head`

注意：这组 run 的 `quick_validation_conclusion.json` 中 `eligible_run_count=0`，原因包括 `target_path_descriptor_supervision`、`target_path_label_supervision`、`target_radio_label_supervision`、`run_marked_ineligible` 和 `split_eligibility_unknown`。因此它们适合作为 v8 target-prior 机制诊断，不应直接写成 main conclusion。

source-only 基线在这组配置下为：

| 方案 | Top-1 | Top-3 | Top-5 | MAE | Beam loss | Within-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source-only | 0.0146 | 0.0245 | 0.0286 | 16.3386 | 6.2407 | 0.0654 |

A2-A5 的 adapted target test 结果：

| 方案 | mode | Top-1 | Top-3 | Top-5 | MAE | Beam loss | Within-3 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A2 | `target_linear_probe` | 0.0782 | 0.1144 | 0.1349 | 15.3923 | 6.0163 | 0.2837 | Top-1 / beam loss 最好，但 Top-3/5 较弱 |
| A3 | `target_prior_head` | 0.0636 | 0.1407 | 0.2020 | 15.2446 | 6.0644 | 0.3123 | MAE / Within-3 最好 |
| A4 | `source_prior_only` | 0.0146 | 0.0245 | 0.0286 | 16.2995 | 6.2523 | 0.0654 | 基本退回 source-only |
| A5 | `target_prior_coarse_to_fine` | 0.0630 | 0.1413 | 0.2026 | 15.4127 | 6.0718 | 0.3106 | Top-3/5 略高于 A3，但差距极小 |

逐 horizon 看，A2 的收益主要来自后两个 horizon，而 A3/A5 更偏向第一个 horizon 和 Top-3/5：

| 方案 | Top-1 t1 | Top-1 t2 | Top-1 t3 | Top-3 t1 | Top-3 t2 | Top-3 t3 | Top-5 t1 | Top-5 t2 | Top-5 t3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 | 0.0000 | 0.1103 | 0.1243 | 0.0701 | 0.1401 | 0.1331 | 0.0911 | 0.1541 | 0.1594 |
| A3 | 0.1016 | 0.0823 | 0.0070 | 0.1699 | 0.1611 | 0.0911 | 0.2802 | 0.2277 | 0.0981 |
| A4 | 0.0123 | 0.0158 | 0.0158 | 0.0263 | 0.0245 | 0.0228 | 0.0298 | 0.0280 | 0.0280 |
| A5 | 0.0963 | 0.0858 | 0.0070 | 0.1751 | 0.1576 | 0.0911 | 0.2837 | 0.2259 | 0.0981 |

结论上，A3 是当前最合理的默认 v8 mode；A5 没有证明 coarse-to-fine 带来实质增益；A2 可作为 Top-1 偏好的 ablation；A4 证明 target prior 对这组 target shift 是必要项。

## 7. 预测分布观察

### 7.1 原始 multisource v7

预测高度集中在：

- `33`
- `34`
- `35`

而 target 真值主要集中在：

- `47`
- `48`
- `49`
- `50`
- `52`
- `54`
- `55`

这是典型的 source prior collapse。

### 7.2 residual_free 之后

`residual_free` 把预测从 `33/34` 拉到了 `51/52` 一带，说明 residual 确实开始朝 target 区域移动了，但仍然过粗。

### 7.3 residual 正则调小之后

`v7_res_l2=1e-4` 和 `1e-3` 都比 `0` 更稳，说明：

- residual 不能完全放飞
- 但 `0.01` 级别的默认正则又太强

当前更像是一个“需要很小但不能没有”的正则点。

### 7.4 v8 A2-A5 的分布塌缩

v8 A2-A5 确实能把预测从 source-only 的错误区间拉向 target 高频 beam，但预测分布仍然过窄。target test 真值 top-5 beam 只覆盖约 41.9% 样本，而 v8 预测高度集中：

| 方案 | 预测 top-1 占比 | 预测 top-2 占比 | 预测 top-5 占比 | 预测非零 beam 数 | 主要预测 beam |
| --- | ---: | ---: | ---: | ---: | --- |
| A2 | 0.592 | 0.849 | 0.976 | 15 | `48`, `35`, `12`, `4`, `29` |
| A3 | 0.336 | 0.661 | 0.999 | 6 | `48`, `46`, `34`, `47`, `4` |
| A4 | 0.735 | 0.968 | 0.992 | 8 | `34`, `33`, `35`, `37`, `43` |
| A5 | 0.326 | 0.651 | 1.000 | 5 | `48`, `46`, `34`, `47`, `4` |

这说明 v8 A3/A5 的提升主要来自 few-shot target prior 把输出拉到 target 主峰附近，而不是恢复完整 beam 分布。A4 仍停留在 source prior 区间，进一步确认单靠 source prior 无法解决 crossroad shift。

## 8. 初步结论

1. 当前 multisource v7 不是简单的“没训好”，而是 source prior 在 target crossroad 上明显偏到错误 beam 区间。
2. `target_balance` 单独作用很弱，几乎不能改变主预测簇。
3. `v7_res_l2` 是最关键旋钮。
4. 当前最好的 v7 probe 仍然没有超过单源 fixed-source baseline。
5. `v6_full_finetune` 在这组设置下是负面结果，不适合作为主解法。
6. v8 target-prior head 能显著强于同配置 source-only，其中 A3/A5 在 Top-3/5 和 MAE 上最有价值。
7. v8 A3/A5 仍存在严重预测集中，当前更像是“target prior 校正器”，不是完整的跨场景泛化解法。
8. v8 A2-A5 使用了 target path/radio 监督，被 quick validation 标为 main conclusion 不合格；它们只能作为诊断或 ablation 证据。

## 9. 当前最大问题

### 9.1 source prior 塌缩

多源 source_train 本身就把模型带偏了，预测集中到 source 高频 beam 区间，而不是 target 的主峰。

### 9.2 adaptation 自由度太小

target adaptation 只调 very small subset of parameters，且没有 target physical oracle，纠偏空间不足。

### 9.3 residual 正则过强

默认 `v7_res_l2=0.01` 很可能把 residual 压得太死，导致模型更像“保守地坚持 source prior”，而不是补偿 target shift。

### 9.4 top1 与 power 指标存在 trade-off

`residual_free` 和低正则 probe 能明显改善 Top-3 / NRP，但 Top-1 仍然不够高，说明模型更容易学到“邻域正确”而不是“精确 beam 正确”。

### 9.5 target prior 能纠偏但容易过窄

v8 A3/A5 说明少量 target label 可以把主预测簇从 source 区间拉回 target 高频区间，但输出 beam 种类很少。后续如果继续走 target-prior 路线，需要解决 prior 过强、分布熵过低和长尾 beam 覆盖不足的问题。

## 10. 建议下一步讨论的问题

1. 是否需要放开 shared head 或 transformer 后几层，而不是只训练 private residual？
2. 是否应把 `v7_res_l2` 作为可调主参数，默认下调到 `1e-4 ~ 1e-3`？
3. multisource source_train 是否要重新看 source scene balance / beam histogram？
4. target 10-shot 是否太少，是否需要更细粒度的 few-shot 策略？
5. 是否要把 `top1` 和 `NRP` 作为联合目标来找折中，而不是只看单一分类准确率？
6. v8 A3 是否应作为 target-prior 默认模式，A5 只作为 coarse-to-fine ablation 保留？
7. 是否需要给 v8 增加 prior temperature、entropy regularization 或更弱的 prior mixing，避免预测只落在 5-6 个 beam 上？
8. 如果要产出 main conclusion，是否需要跑不使用 target path/radio supervision 的 v8 变体，或把这组结果明确标注为 ineligible diagnostic？

## 11. 历史实现线索

这些文件在当前源码树中已经退役。需要复核旧实现时，请通过 git 历史或归档 OpenSpec 查找原 `src/kd_sensing/models/fusion/hist_beam.py`、`src/kd_sensing/engine/hist_beam_*`、`configs/hist_beam/*` 和相关 change 记录；当前开发不应新增对这些路径的依赖。
