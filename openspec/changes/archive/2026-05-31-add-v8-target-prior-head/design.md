## Context

当前 MMW sensor-assisted quick validation 显示，`v7_shared_physical_private_residual` 的 source/shared logits 在 target crossroad 上集中预测 source 高频 beam 33/34/35，而 target 真值主峰位于 47/48/49/50/52/54/55。v7 adaptation 默认只训练 `private_adapter`、`private_residual_head` 和 `residual_gate`，并且最终预测仍以 frozen `logits_shared` 为基底，因此 few-shot residual 很难推翻已经塌缩的 source prior。

现有代码中 HiST-Beam 模型、loss、target adaptation 和 LOSO stage 已经有 v7 分支和 trainable ratio 记录。本变更应沿用这些入口，新增一个最小可验证的 v8 分支，避免重构训练框架或改变旧 v0-v7 行为。

## Goals / Non-Goals

**Goals:**

- 新增 `v8_target_prior_head`，默认使用 `target_logits + beta_prior * target_prior_bias` 作为最终 beam logits。
- 冻结 source backbone 后，只训练 target-specific adapter/head/prior 及可选 coarse-to-fine 诊断头，验证 frozen representation 是否包含 target 可分信息。
- 仅使用 target_adapt labeled support labels 初始化 target prior，不使用 target_test label 或 target-side physical/radio/path oracle。
- 为 v8 target adaptation 使用 beam topology soft label、prior smoothness 和可选 sector/offset loss。
- 输出 prediction histogram 诊断，直接观察 source prior collapse 是否缓解。
- 保持 v7、history-anchor 和 path-prototype 变体默认行为不变。

**Non-Goals:**

- 不在第一阶段实现完整 budget/seed sweep 或替代现有 quick validation planner。
- 不把 target test label、beam_power、path params、CSI 或 radio label 引入 adaptation 训练。
- 不删除或重命名 v7，也不把 v8 作为默认 HiST-Beam 变体。
- 不优先实现 source train long-tail 去偏 loss；本阶段只保留配置入口或 TODO，除非实现成本很低且不会影响旧配置。

## Decisions

### Decision 1: v8 默认不使用 source logits 参与 final prediction

v8 forward 默认输出：

```python
final_logits = target_logits + beta_prior * target_prior_bias
```

`logits_shared` 或 source/shared beam logits 继续作为诊断输出保留，但 `hist_beam.v8.use_source_logits_in_final=false` 时不参与 `logits`、`beam_logits` 或 `logits_final`。

理由：当前问题是 source logits 本身塌缩，继续把 source logits 作为主项会把 v8 退化成另一个 residual 修正实验。保留可选融合：

```python
final_logits = lambda_src * source_logits_debiased + lambda_tgt * target_logits + beta_prior * target_prior_bias
```

用于 A4 诊断，但默认 `lambda_src=0.0`。

### Decision 2: target prior 在 adaptation 前由 support labels 写入模型

模型提供 `set_target_prior_from_labels(labels, sigma=1.5, eps=1e-4)`。函数将 target_adapt labeled support labels 转换为 Gaussian-smoothed histogram，写入 `target_prior_bias`，并记录 support label hist、smoothed prior top beams 和 bias top beams。

理由：模型初始化阶段拿不到 few-shot subset；在 target adaptation stage 构造 labeled loader/sampling manifest 后再初始化 prior 更符合现有 LOSO stage 边界，也更容易审计防泄漏。

### Decision 3: v8 使用独立 target adapter/head，而不是复用 v7 private residual head

新增 `target_adapter = BottleneckAdapter(d_model, adapter_dim, dropout)` 和 `target_head = nn.Linear(d_model, num_pred * num_classes)`。`use_adapter=false` 时 target head 直接读取 `fused`，用于 target linear probe。

理由：复用 v7 residual head 会继续把建模目标表达成 source logits 的 delta，不利于隔离“frozen backbone 是否可线性分离 target beam”这个核心问题。独立 head 也便于 freeze policy 精确控制 trainable 参数。

### Decision 4: v8 loss 放在现有 HiST-Beam loss 入口内分支处理

`compute_hist_beam_loss` 先识别 v8，再调用 `_compute_v8_hist_beam_loss`。v8 target adaptation 默认使用：

- final soft CE：基于 hard beam label 生成 Gaussian beam topology soft labels；
- prior smoothness：`mean((bias[1:] - bias[:-1]) ** 2)`；
- 可选 sector CE；
- 可选 offset CE。

source training 阶段若运行 v8，可以先使用 final hard CE 或与 target 相同的可配置 soft CE，但默认不读取 target-side physical oracle。

理由：现有 adaptation loop 已统一调用 `compute_hist_beam_loss`，在 loss 层新增 v8 分支比新建训练循环更小侵入。

### Decision 5: freeze policy 通过现有 `apply_hist_beam_adaptation_strategy` 扩展

新增策略 `v8_target_head_only`，冻结 encoders、feature projections、transformer、shared/private source branches、source/shared beam head 和 physical head，只训练：

- `target_adapter`
- `target_head`
- `target_prior_bias`
- `beta_prior`，当 `learnable_beta_prior=true`
- `sector_head` / `offset_head`，当 `use_coarse_to_fine=true`
- 可选 LayerNorm/BN affine，沿用 `train_layernorm_affine`

`unfreeze_last_fusion_block=true` 是显式 opt-in，且配置或日志必须能显示低学习率意图。

### Decision 6: prediction histogram 作为标准 eval artifact

在 source-only target eval 和 adapted target eval 后写出 `prediction_hist.json`，并把 `true_top_beams`、`pred_top_beams`、`mean_abs_beam_error`、`within_1_acc`、`within_2_acc`、`within_3_acc` 合并到 metrics 或 summary 可读取位置。

理由：本次 v8 的主要假设是修正预测分布，不只追 Top-1；histogram artifact 是判断 source prior collapse 是否缓解的最低成本证据。

## Risks / Trade-offs

- [Risk] 10-shot support prior 过强，模型可能只复刻 few-shot label histogram。  
  Mitigation: 保留 A2 target linear probe、A4 source prior only 和 `beta_prior` 可学习/可固定配置，日志输出 prior top beams 与 prediction top beams 便于区分。

- [Risk] `target_prior_bias` 可训练后可能过拟合极少数 support labels。  
  Mitigation: 默认启用 `loss_prior_smooth_weight=0.001`，并保留 `learnable_beta_prior` 和 `beta_prior` 配置开关。

- [Risk] coarse-to-fine offset head 在 `num_classes % sector_size != 0` 时最后一个 sector 有越界 offset。  
  Mitigation: loss 中对非法 beam/offset 做 mask；若实现选择先要求可整除，必须在配置解析阶段清晰报错。

- [Risk] v8 与历史 `v8_path_proto` 名称可能造成语义混淆。  
  Mitigation: 新 variant 使用完整名称 `v8_target_prior_head`，不复用 `v8_path_proto`，metadata 中记录 `v8_target_prior_head=true`。

- [Risk] prototype classifier 诊断实现成本高。  
  Mitigation: 第一阶段可保留清晰接口和 TODO；若 `run_prototype_probe=true` 但未实现，必须在 metrics 中记录 unavailable reason，而不是静默跳过。

## Migration Plan

1. 添加 v8 配置解析与模型分支，默认配置不启用 v8。
2. 添加 prior 初始化工具和模型 setter，在 target adaptation stage few-shot sampling 后调用。
3. 添加 v8 loss、freeze policy 和诊断日志。
4. 添加 `configs/hist_beam/v8_target_prior_head.yaml` 或 quick validation override 示例。
5. 添加 histogram artifact 写出，并让 LOSO summary 能引用。
6. 使用 `conda run -n kd_mm_beam pytest ...` 跑 v8 相关测试和现有边界测试；使用 `openspec validate add-v8-target-prior-head --strict` 校验变更。

回滚策略是关闭配置中的 `hist_beam.variant=v8_target_prior_head`，旧 v7 和其它变体不依赖 v8 参数。
