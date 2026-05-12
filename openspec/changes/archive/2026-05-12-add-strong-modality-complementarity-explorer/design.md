## Context

现有弱模态互补分析已经实现：

- `build_complementarity_cases.py` 从 Conditional Utility Audit 产物读取 `subset_predictions`、`teacher_predictions`、per-sample delta 和 bucket 表。
- `complementarity.py` 默认把 `strong_only` 解析为 `gps+mmwave`，把 `image/radar/lidar` 作为 weak modalities，并对齐 `strong_only`、weak teacher / weak subset、`strong_plus_<weak>` 三类预测。
- Gradio `Complementarity Explorer` 当前只有 `Weak Modality` 筛选控件，前端 helper 只按 `weak_modality`、horizon、case/tag、bucket、gain 和 sort 过滤。

用户现在要回答的是“某个强势模态，例如 `mmwave` 或 `gps`，和一个或全部 weak modalities 的互补关系”。这和现有 `strong_only=gps+mmwave` baseline 不完全等价：强势模态 anchor 应该能是单模态预测源，fusion subset 也可能不是现有 `strong_plus_<weak>`。当前 Conditional Utility Audit 已经可以 dump 单模态 teacher predictions，Scene32 配置通常会包含 `gps`、`mmwave`、`image`、`radar`、`lidar` teacher 行；但 subset predictions 默认只覆盖 `strong_only`、`strong_plus_image/radar/lidar`、`single_best_mmwave` 和 `weak_only`，不保证存在 `gps+image` 或 `mmwave+image` 这类 strong+weak fusion subset。

## Goals / Non-Goals

**Goals:**

- 在不破坏旧默认行为的前提下，为互补 case 表新增 `strong_modality` 维度。
- 支持用户通过 CLI 选择一个或多个 strong modalities，并通过 Explorer 选择单个 strong modality 或 `all`。
- 对每个 strong/weak pair 输出强势模态预测、弱模态预测、互补标签、概率增益和 prediction source。
- 在缺少 strong+weak fusion subset 时仍能分析 strong-vs-weak 互补关系，并让 fusion/rescue 指标显式不可用。
- 扩展 summary、导出、详情 JSON 和测试，使结果按 strong modality 与 strong/weak pair 可解释。

**Non-Goals:**

- 不重新定义哪些模态是强势模态；默认只面向当前研究中的 `gps`、`mmwave`，但实现允许配置覆盖。
- 不要求本次变更新增 Conditional Utility Audit 的推理 subset，也不重新跑模型。
- 不修改 MARF、CRAF、G2D 或 teacher 模型结构。
- 不从 top-k 概率伪造完整 64 类分布；完整分布仍由 viewer manifest 的 `beam_distribution` 决定。

## Decisions

### Decision 1: 增加可选 strong modality mode，保留旧 strong subset mode

`build_case_table()` 保留现有参数和行为：未传 `strong_modalities` 时继续使用 `strong_subset=strong_only`，输出旧式 weak complementarity case。新增参数建议为：

- `strong_modalities: Iterable[str] | None`
- `strong_prediction_sources` 或内部 metadata，用于记录每个 strong modality 的来源
- `pair_fusion_subsets: dict[str, str] | None`，key 使用 `strong+weak` 或等价 pair key

当 `strong_modalities` 非空时，后端进入 pair mode：循环 `strong_modality × weak_modality`，strong prediction 从单模态预测源解析，weak prediction 沿用现有 `_select_weak_predictions()` 语义，fusion prediction 变成可选。

理由：旧分析已经用于当前报告和 Explorer，直接把 `strong_subset` 改成强势模态会引入兼容风险。显式 pair mode 让新功能可以逐步接入，同时旧命令不变。

替代方案是把 `strong_subset` 泛化为任意单模态或 subset。该方案 API 表面更少，但会混淆 “strong_only subset baseline” 和 “single strong modality anchor” 两种研究语义。

### Decision 2: 预测来源选择复用 normalized schema

后端继续通过 `normalize_schema()` 统一 `subset_predictions` 和 `teacher_predictions` 字段。新增一个通用 helper，例如 `_select_modality_predictions(subset, teacher, modality, role)`：

1. 先查 `subset["subset_key"] == modality` 或别名，例如 `single_best_mmwave` 可解析为 `mmwave`。
2. 再查 `teacher["teacher_modality_key"] == modality` 或 `teacher["subset_key"] == modality`。
3. 如果都缺失，返回空 DataFrame 和 source metadata。

该 helper 可被 strong 和 weak 两侧复用，但 weak 侧保留现有函数包装，避免影响旧测试命名。

理由：当前 Scene32 的单模态强势预测更可能来自 `teacher_predictions`，而 `single_best_mmwave` 又可能出现在 subset predictions。统一来源选择能覆盖两种情况，并把 provenance 写入 summary。

替代方案是强制 strong modality 只来自 teacher predictions。该方案简单，但会忽略已有 `single_best_mmwave` subset，也不利于未来 dedicated single-modality subset 输出。

### Decision 3: fusion subset 是 pair-level optional input

pair mode 不假设 `strong+weak` fusion subset 一定存在。解析策略：

- 如果用户提供 `--pair-fusion-subsets mmwave+image=mmwave_plus_image,...`，优先使用显式映射。
- 否则尝试规范化候选名：`<strong>+<weak>`、`<strong>_<weak>`、`strong_plus_<strong>_<weak>` 等。
- 找不到时仍输出 case，设置 `fusion_prediction_available=false`，并让 `fusion_pred`、`fusion_correct`、`fusion_gt_gain`、rescue / unused / negative transfer 相关主类为空值或不可用。

强弱互补的核心标签 `strong_wrong_weak_correct` 只依赖 strong 和 weak top1，因此在 fusion 缺失时仍可计算。旧 `strong_only` mode 仍要求 `strong_plus_<weak>`，保持现有 rescue 语义。

理由：当前 audit outputs 不保证存在 pair fusion subset。如果把 fusion subset 作为硬依赖，用户最关心的 strong-vs-weak 关系会无法查看。

替代方案是自动复用现有 `strong_plus_<weak>` 作为所有 strong modality 的 fusion。该方案会把 `gps+mmwave+weak` 误解释成 `mmwave+weak` 或 `gps+weak`，研究语义不准确。

### Decision 4: summary 新增 strong 维度，但全局统计保持兼容

`compute_summary()` 继续输出现有 `global`、`by_weak_modality`、`by_horizon` 和 `by_case_type`。当 case 表包含 `strong_modality` 时，额外输出：

- `by_strong_modality`
- `by_strong_weak_pair`
- metadata 中的 `strong_prediction_sources`
- metadata 中的 `fusion_subset_availability`

对于 fusion unavailable 的 group，strong/weak 互补率继续输出，fusion-dependent rate 使用空值，并记录 unavailable reason。这样旧消费者仍能读取原字段，新 Explorer 可以读取新字段。

理由：summary 是下游报告和 UI 的公共接口，新增字段比重命名字段更稳妥。

替代方案是新增一套独立 `strong_modality_complementarity_summary.json`。该方案隔离性强，但会让 viewer 同时支持两套 summary 格式，维护成本更高。

### Decision 5: Explorer 只新增一个强势模态控件和 helper 参数

`build_complementarity_choices()` 在 case 表包含 `strong_modality` 时返回：

- `strong_modalities`
- `defaults["strong_modality"]`

`filter_complementarity_cases()` 新增 `strong_modality` 参数，并在现有 weak modality filter 前后应用均可。Gradio 页面在 `Scene/Horizon/Weak Modality` 同一行加入 `Strong Modality`，默认选择 `mmwave`，没有 `mmwave` 时选择第一个可用值。若 case 表不含 `strong_modality`，choices 退化为 `["all"]`，旧输出仍可加载。

理由：用户要求“模仿 Weak Modality 添加一个强势模态”，最直接的交互就是同级 dropdown。helper 层先扩展，Gradio 回调只增加一个输入输出路径，测试也清晰。

替代方案是把 strong/weak pair 做成一个组合 dropdown。该方案能减少控件，但不方便表达“某个强势模态 + 全部 weak modality”。

## Risks / Trade-offs

- [Risk] 当前 audit 没有 `gps+weak` 或 `mmwave+weak` fusion subset，导致用户看不到 rescue 指标。→ Mitigation：pair mode 把 fusion 做成可选，UI 和 summary 显式标记 unavailable，同时继续输出 strong-vs-weak 互补率。
- [Risk] `single_best_mmwave`、`mmwave`、`teacher_mmwave` 等别名解析不一致。→ Mitigation：把别名集中到 `canonical_subset_name()` 或新增 pair 解析 helper，并用单元测试覆盖。
- [Risk] 旧 case 表没有 `strong_modality`，新增 Explorer 控件可能破坏旧输出加载。→ Mitigation：choices 和 filter 对缺失列退化为 `all`，旧 test fixture 保持可用。
- [Risk] 全局统计在 strong/weak pair mode 下会把多个 pair 的同一个样本重复计入。→ Mitigation：summary metadata 明确统计单位是 case row；pair 级结论优先看 `by_strong_weak_pair`。
- [Risk] prediction source 混用 teacher 和 subset，可能造成研究解释偏差。→ Mitigation：case 表和详情 JSON 固定展示 `strong_prediction_source` 与 `weak_prediction_source`，报告中保留来源 metadata。

## Migration Plan

1. 扩展后端 helper 和 tests，先保证旧 `build_case_table()` 调用不变。
2. 增加 CLI 参数并用人工 DataFrame / 小 fixture 验证 pair mode。
3. 扩展 Explorer helper 和 Gradio 控件，确保旧 complementarity 输出仍能 check-only 加载。
4. 更新 README，说明旧 `strong_only` 分析和新 strong modality pair 分析的命令差异。
5. 回滚方式：不传 `--strong-modalities` 即保持旧分析；Gradio 若加载旧 case 表会把 `Strong Modality` 退化为 `all`。

## Open Questions

- 是否需要在后续单独扩展 Conditional Utility Audit subset registry，生成真实 `gps+image`、`mmwave+image` 等 fusion subset。当前设计先不强依赖这些产物。
- 默认 strong modalities 是否固定为 `gps,mmwave`，还是从 teacher registry / summary 自动按 prior 或 performance 选 top-k。当前设计采用配置优先、Scene32 默认 `gps,mmwave`。
- `single_best_mmwave` 是否应规范化为 `mmwave` 并优先于 teacher predictions。实现时可通过测试固定最终优先级。
