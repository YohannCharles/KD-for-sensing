## Context

`add-v8-target-prior-head` 已完成并跑出 A2-A5 诊断结果。结果表明 frozen source backbone 仍包含一部分 target 可分信息，source head/source logits 是主要负迁移来源；但 A3/A5 的预测高度集中到 target support 高频 beam，说明 global target prior 过强，模型尚未学到输入条件化的 target 判别。

同时，当前 v8 quick validation summary 出现 `eligible_run_count=0`。下一步实验若不先修正 eligibility 判定，就只能作为机制诊断，不能作为正式主结论。v9 的设计因此分为两层：P0 先审计和修正 run eligibility；P1/P2 在 v8 基础上加入 collapse 诊断与 input-conditioned local calibration。

## Goals / Non-Goals

**Goals:**

- 让 MMW sensor-assisted quick validation 的 eligibility 判定基于实际 split 防泄漏证据和实际使用的 target-side oracle 字段。
- 为 v8/v9 输出足够的 collapse 来源诊断，区分 global prior、target head 和 prototype/local calibration 的贡献。
- 新增 `v9_input_conditioned_target_adaptation` 或等价 v8 mode，默认 final logits 为：

```python
final_logits = target_logits + beta * global_target_prior_bias + eta * prototype_logits
```

- 支持 beam prototype 与 sector prototype，优先验证 `sector_size=2/3`，不用 A5 的 `sector_size=4` 作为主线。
- 限制 global prior 的塌缩能力：beta cap、fixed beta、prior dropout 和 widened-prior marginal KL。
- 保留 v8 A2/A3/A4/A5 作为 baseline/ablation，不改变旧配置默认行为。

**Non-Goals:**

- 不在本变更中重训 source backbone 或默认启用 Balanced Softmax/Logit Adjustment。
- 不引入 target_test label、target_test beam_power、target_test path/radio fields 参与 adaptation、early stopping、prior 初始化、temperature fitting 或 prototype 更新。
- 不把 pseudo-label CE 作为默认无标签训练策略；若使用 target unlabeled，第一阶段只做分布约束、consistency 或 confidence cap。
- 不把 coarse-to-fine head 继续扩成主方法；A5 保留为 ablation。
- 不新增外部依赖。

## Decisions

### Decision 1: 先修 eligibility，再跑 v9 主实验

实现前先审计 current quick validation 被标记为 ineligible 的原因，并把原因拆成至少两类：

- split 或 history/window 防泄漏证据不足；
- target path/radio/oracle 字段实际参与 adaptation 或选择；
- validator 规则过严，错误要求了未使用的 path/radio supervision 条件。

run metadata 和 summary 必须记录 `eligibility_status`、`eligibility_reasons`、`used_target_oracle_fields`、`target_oracle_usage_stage` 和 split diagnostics。只有 split strict 且未使用禁用 oracle 字段的 run 才能进入主结论。

理由：v9 方法指标即使提升，如果仍被 summary 归为 ineligible，也无法支撑论文主线。

### Decision 2: collapse diagnostics 作为 v9 前置产物

在 adapted eval 后写出 `collapse_diagnostics.json` 或等价 metrics 字段，至少包含：

- `support_prior_hist`、`true_hist`、`pred_hist`；
- `kl_pred_support`、`kl_true_support`、`kl_pred_true`；
- `unique_pred_beams`、`pred_top_beams`、`true_top_beams`；
- `beta_prior_initial`、`beta_prior_final`、`beta_prior_effective`；
- `target_logits_only`、`prior_only`、`target_logits_plus_prior` 的 Top-K/within3/MAE/hist；
- per-true-beam confusion，覆盖 target 高频 beam 47/48/49/50/52/54/55 或当前 target 真值 top beams。

理由：这些诊断能判断 A3 collapse 是由 prior 复制、target head 过拟合还是两者共同导致。

### Decision 3: global prior 只做受限粗校准

v9 不允许 `beta_prior` 无界增长。可训练 beta 使用：

```python
beta = beta_max * sigmoid(raw_beta)
```

并支持 `beta_prior_trainable=false` 的 fixed beta ablation。训练时可启用 prior dropout：

```python
if training and rand() < prior_dropout:
    prior_term = 0
else:
    prior_term = beta * target_prior_bias
```

理由：A3 已证明全局 prior 有用，但全局 prior 对所有样本相同，天然会把分布吸向 support 高频 beam。

### Decision 4: prototype logits 提供输入条件化的 local calibration

target support features 由 frozen backbone 或 v9 target adapter 提取，按 beam 或 sector 聚合 prototype：

```python
prototype_c = mean(normalize(feature_i)) for y_i == c
prototype_logits[c] = cosine(normalize(feature), prototype_c) / tau
```

缺失 prototype 的 beam/sector 使用 masked logits 或 fallback 到平滑邻域 prototype，不得填充为高置信伪 prototype。final logits 默认组合 target logits、受限 prior 和 prototype logits：

```python
final_logits = target_logits + beta * target_prior_bias + eta * prototype_logits
```

sector prototype 需要把 sector logits 映射回 beam logits，可以用同 sector beam 共享 prototype 分数，或结合 beam topology smoothing 分配给邻近 beam。实现必须在 metadata 中记录映射方式。

理由：prototype logits 直接依赖 query feature 与 support feature 的相似度，比 global prior 更能恢复样本条件差异。

### Decision 5: anti-collapse loss 匹配 widened target prior，而不是 uniform

若启用 marginal regularization，构造比 support prior 更宽的 `widened_target_prior`，例如更大的 Gaussian sigma 或 temperature。loss 使用 batch prediction marginal：

```python
p_bar = softmax(final_logits).mean(dim=0)
loss_dist = KL(p_bar || widened_target_prior)
```

该 loss 默认低权重，只用于避免预测集中到 1-2 个 beam；不得强制预测均匀化。

理由：target 真实分布本来不均匀，uniform entropy 会把物理 beam 分布拉向错误目标。

### Decision 6: 实验矩阵小而可解释

默认 v9 quick validation 只生成三组小矩阵：

- Group A：A3-base、A3-no-prior、A3-fixed-beta、A3-prior-dropout；
- Group B：P1 beam prototype only、P2 sector prototype only、P3 A3+beam prototype、P4 A3+sector prototype；
- Group C：可选 U1 A3+widened prior KL、U2 A3+prototype+widened prior KL。

Group C 只有在 protocol 允许使用未标注 target_adapt 且 metadata 能证明未读取 target_test 时启用。

## Risks / Trade-offs

- [Risk] prototype support 太少，beam-level prototype 噪声很大。  
  Mitigation: 同时支持 sector prototype 和 beam prototype；记录 per-class support count，低 count prototype 使用 mask/fallback。

- [Risk] anti-collapse loss 可能降低 Top-1。  
  Mitigation: 验收指标同时关注 Top-3、Top-5、within3、MAE 和 pred coverage；低权重默认，作为 ablation 而非强制主配置。

- [Risk] eligibility 修正可能发现已有 v8 run 确实使用了禁用 oracle。  
  Mitigation: 将这些 run 标记为 excluded/debug，移除 oracle 路径后重跑；不通过改 summary 把不合格 run 包装成主结论。

- [Risk] v9 mode 与 v8 配置重叠导致维护复杂。  
  Mitigation: 尽量复用 v8 target branch、prior 初始化、loss 和 histogram artifact；只新增 v9 mode/配置项和 prototype/prior-control 分支。

- [Risk] prototype logits 与 target prior 同时启用后仍塌缩。  
  Mitigation: 必须保留 `prototype only`、`target_logits only`、`prior only` ablation，并在 collapse diagnostics 中输出各分支独立指标。

## Migration Plan

1. 添加或修正 eligibility checker，先让现有 v8 run 的 ineligible reason 可解释。
2. 在 v8 eval/adapted eval 后新增 collapse diagnostics，不改变模型行为。
3. 新增 v9 配置解析、beta cap/fixed beta/prior dropout 和 prototype logits。
4. 新增 widened-prior marginal KL 与可选 consistency/dropout 约束。
5. 新增 v9 quick validation override 和 summary 聚合字段。
6. 运行定向测试、`openspec validate add-v9-input-conditioned-target-adaptation --strict`，再跑小矩阵验证。

回滚策略是关闭 `hist_beam.variant=v9_input_conditioned_target_adaptation` 或对应 v8 mode，并关闭 v9 diagnostic/loss 配置；旧 v8 与更早变体不依赖 v9 参数。

## Open Questions

- `prototype_logits` 默认使用 frozen fused features 还是经过 target adapter 的 features，需要在首轮 P1/P3 对比后固定。
- sector prototype 回填到 beam logits 的默认映射采用 sector 内均分还是 beam topology smoothing，需要由实现复杂度和诊断清晰度决定。
- 当前 protocol 是否允许使用未标注 target_adapt 做 Group C；如果 metadata 无法证明，则 Group C 默认禁用。
