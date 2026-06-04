## Context

DeepSense6G scenario31-34 已有 GPS-only v2 adapter 配置与运行产物，当前 r15 target-adapt sweep 的 overall 结果约为 `DBA=0.6707`、`mean err=2.344`、`P(error<4)=0.8523`。这说明 GPS prior 已经是强 anchor，后续 residual workflow 应围绕“少量 hard residual 样本纠偏”设计，而不是训练一个多模态模型从零覆盖 GPS 预测。

现有仓库要求新代码位于 `src/kd_sensing/` 包结构内，入口通过包内 CLI 或 console script 暴露。因此需求文本中的 `python -m src.*` 命令在实现时应改写为 `python -m kd_sensing.cli.*` 或 `kd-sensing-*` 命令，参数、输出目录和功能保持等价。

## Goals / Non-Goals

**Goals:**

- 以 GPS v2 predictions/logits 为 coarse prior，构建 residual correction、gate 和 top-K re-rank workflow。
- 默认支持 scenario31-34、`mapping_disabled`、64 beam circular labels、`support_ratio=0.15` 和 `target_adapt_beambench_residual`。
- 至少保证 `gps_prior_only` 与 `gps_context_only_residual` 能在没有 image/LiDAR/radar 资源时运行。
- 在 image/LiDAR/radar path 或预计算 feature 可用时自动启用对应 ablation；不可用时跳过并记录原因。
- 所有指标与诊断都必须和 GPS v2 baseline 对齐比较，并报告 hard sample correction 与 good sample degradation。
- 保证 query label 只用于最终评价、图和报告，不参与模型选择、early stopping、support 构造、prior 构造或调参。

**Non-Goals:**

- 不重写已经完成的 GPS v2 adapter。
- 不把多模态模型从零预测 beam 作为主方法；`modality_only` 只作为反例 ablation。
- 不要求 residual fusion 必然超过 GPS v2；本 change 的成功标准是形成可审计、无泄漏、可比较的 workflow。
- 不下载 pretrained weights，不依赖互联网，不提交 outputs/logs/checkpoint/dataset。
- 不新增顶层 `src.*` 模块或兼容包装入口。

## Decisions

### Decision 1: GPS prior 是模型主锚点

实现使用 `gps_prior_logits` 作为 final logits 的基底：

```text
final_logits = gps_prior_logits + correction_scale * gate * correction_logits
```

`correction_scale` 使用 softplus 参数化并 clamp 到配置的最大值，默认初始化为 0.5、最大 3.0。这样可以让 correction 有表达能力，但训练初期不至于完全覆盖 GPS prior。

替代方案是把 GPS features 与其它模态 concat 后直接分类 64 beams。该方案容易在少量 support 下破坏 GPS already-good 样本，只保留为 `modality_only` 反例。

### Decision 2: prior 来源优先 logits，缺失时用 top1 Gaussian fallback

v2 engine 内部已有 adapter logits 计算逻辑，但现有产物未必保存 `gps_logits.npy`。实现分两层：

- 扩展 v2 运行/输出路径，支持 `--save-logits` 写出 `gps_logits.npy`、`gps_logits_index.csv` 和可选 `gps_prior_probs.npy`。
- residual workflow 检查不到 logits 时，用 `pred_top1` 生成 circular Gaussian prior，并在 manifest、metadata 和 predictions 中标记 `gps_prior_source=fallback_gaussian_from_top1`。

fallback 只使用 GPS prediction，不使用 `target_label`，避免 label leakage。若后续重跑 v2 保存 logits，residual workflow 自动切换到真实 logits source。

### Decision 3: manifest 是 residual workflow 的数据边界

新增 residual manifest 作为 GPS prior、support/query split、GPS context feature、target label、optional modality path/feature 的统一索引。manifest 每行一个样本，并记录 `support_or_query`、`gps_pred_topK`、`gps_error`、`gps_signed_residual`、`gps_is_good_error_lt4`、prior stats 和可用 modality path。

manifest builder 只负责发现和索引，不读取大型模态 tensor。训练 Dataset 再按 manifest 和 enabled modality 读取 image、LiDAR/radar array 或预计算 feature。这样能保持 inspection、manifest、训练和可视化各自职责清晰。

### Decision 4: optional modality 以“可用即用，缺失不阻断”处理

image/LiDAR/radar 的 `enabled: auto` 表示：

- manifest 能发现 path/feature 且数据可读时启用对应 ablation；
- 缺失或 shape 不稳定时跳过该 modality ablation；
- summary 中写入 `skipped_reason`；
- GPS context residual baseline 不受影响。

对于 array modality，优先使用预计算 feature；如果直接读取 map，则必须 shape 可推断且 batch 内可 collate，否则报清晰错误并跳过该 ablation。

### Decision 5: gate 和 good anchor loss 用于保护 good 样本

训练 gate target 来自训练 split 内 `gps_error >= good_error_threshold`。query split 的 good/bad label 只用于最终评估和图。`good_anchor_loss` 仅在 GPS good 样本上计算，使 final distribution 接近 GPS prior；hard 样本仍可由 final CE 和 residual correction 推动。

默认主 ablation 为 `gps_plus_residual_gated_anchor`，但最终推荐方法必须同时看 overall DBA、good sample degradation rate 和 bad sample correction rate。若 DBA 略高但大量破坏 good 样本，不作为推荐。

### Decision 6: 训练协议严格区分 support 和 query

`target_adapt_beambench_residual` 按 target scene 循环：

1. 读取对应 r15 GPS v2 target-adapt artifact。
2. target support 仅来自 v2 support split。
3. target query 仅用于最终评估。
4. source scenes 可用于 residual pretrain；若 source prior predictions 不完整，则降级 `support_only` 并记录。
5. early stopping 和模型选择只能使用 source validation 或 target support 内部划分，不得读取 target query label。

`within_scene_residual_upper_bound` 仅作为上界，不进入主结论；`gps_prior_only` 用于复现 v2 baseline 和 sanity check。

### Decision 7: 输出契约先服务诊断

主输出目录为：

```text
outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled/
```

必须写出 `summary_overall.csv`、`summary_by_scene.csv`、`summary_by_gps_good_bad.csv`、`predictions.csv`、`correction_events.csv`、`candidate_recall.csv`、`comparison_report.md` 和 figures。所有 summary 都包含 GPS baseline 指标与 residual delta 字段，使失败结果也能解释“为什么没超过”。

## Risks / Trade-offs

- [Risk] 现有 v2 产物没有 logits，Gaussian fallback 会弱化 prior fidelity。→ Mitigation：inspection 明确报告 prior source，并提供 `--save-logits` 扩展以便重跑 v2。
- [Risk] optional modality 文件路径在不同 DeepSense6G 场景中不一致。→ Mitigation：manifest builder 先做自动发现和 warning，训练只启用可稳定读取的 modality。
- [Risk] small support 下 residual model 过拟合并破坏 GPS good 样本。→ Mitigation：gate、good anchor loss、correction scale clamp、good degradation rate 和 gated/no-gate ablation 联合诊断。
- [Risk] query leakage 难以人工审计。→ Mitigation：split role 写入 manifest、run metadata 记录 model_selection_split，测试覆盖 fallback prior 和 gate target 不使用 query label。
- [Risk] top-K reranker candidate recall 不足导致 rerank loss 样本少。→ Mitigation：报告 `target_in_gps_top16`、`target_in_local_radius8`、`target_in_union_candidates`，且 rerank 不是第一版性能硬门槛。

## Migration Plan

1. 新增 OpenSpec 规格与 README 说明，确认 scope。
2. 实现 circular utility、prior loading、input inspection 和 manifest builder。
3. 扩展 GPS v2 输出 logits 能力，但保持已有 v2 默认结果兼容。
4. 实现 residual fusion model、encoders、losses、Dataset/DataLoader 和 train/eval loop。
5. 实现 top-K reranker、plotter、baseline comparison report 和 summary outputs。
6. 运行新单测、OpenSpec 校验和 residual workflow 验收命令。

Rollback 方式是删除新增 residual config/CLI/model/loss/tests，并保留现有 GPS v2 adapter 与历史产物不变。

## Open Questions

- 本地 v2 r15 predictions 文件中的 top-K、support/query、timestamp/frame 字段名是否完全满足 manifest 需求，需要 inspection 脚本先确认。
- DeepSense6G scenario31-34 的 image/LiDAR/radar feature/path 是否在当前本地数据中可用，若不可用第一版结果将明确标记为 GPS-context residual。
- 是否为 residual workflow 新增 console scripts，还是仅提供 `python -m kd_sensing.cli.*`；实现时可根据 pyproject 维护成本选择。
