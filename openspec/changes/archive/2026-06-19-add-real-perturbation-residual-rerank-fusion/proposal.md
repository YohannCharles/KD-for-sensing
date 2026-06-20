## Why

`add-geometry-prior-beam-fusion` 已经证明 geometry prior 能守住 clean/P0，但没有明显超过 `Image ResNet+GPS`；同时当前 P0-P5 与 advantage rows 仍来自 deterministic degradation model，不能作为 primary robustness claim。下一步必须先把评估升级为真实扰动 forward，再在强 baseline 之上做受约束的 residual rerank，而不是继续叠普通 late fusion。

## What Changes

- 新增真实扰动 forward evaluation：benchmark runner 对每个 condition 真正应用 difficulty pipeline、执行模型 forward、保存 logits/labels/branch diagnostics，并从真实输出计算 Top-K/DBA。
- 新增 safe residual beam rerank fusion：以强 `Image ResNet+GPS` logits 为 anchor，只在 union top-k candidate set 内使用 geometry prior、reliability 和 branch agreement 做有界 residual/rerank。
- 新增 no-regret clean gate：clean/high-observability 条件下 candidate 不得明显低于 anchor；若 reranker 信心不足或 branch disagreement 高，必须回退 anchor logits。
- 新增 per-sample branch diagnostics：输出 image logits、anchor top-k、geometry prior top-k、rerank candidate set、residual score、fallback reason、branch weights、entropy/agreement 和 selected beam provenance。
- 更新 geometry-prior claim gate：仅真实 per-condition forward 可以升级 primary claim；delegated clean-only 或 synthetic degradation 只能标记 `pending` / `mechanism_diagnostic`。
- 新增 strict 实验矩阵：real P0-P5、GPS advantage slice、clean-only no-regret smoke、reranker ablation、anchor fallback ablation 和 seed=17 strict run。
- 不恢复旧 GPS residual、Top8 selector、camera residual、KD/HiST 或旧根脚本；所有新增模型仍作为 component baseline 或窄诊断 runner 实现。

## Capabilities

### New Capabilities

- `real-perturbation-forward-evaluation`: 定义 benchmark 如何对真实 batch 应用扰动、运行 checkpoint forward、缓存 logits/diagnostics 并生成可升级 claim 的 per-condition metrics。
- `safe-residual-beam-rerank-fusion`: 定义 anchor-safe residual/rerank beam fusion 的 candidate set、bounded residual、fallback、loss、metadata 和 diagnostics 契约。

### Modified Capabilities

- `configurable-multimodal-fusion`: 增加 safe residual/rerank fusion 作为 opt-in component baseline 配置，不新增 whole-model 例外或旧入口。
- `geometry-prior-beam-fusion`: 将 claim gate 升级规则收紧为必须基于真实扰动 forward，并要求 geometry prior diagnostics 能解释 reranker 是否真正使用 prior。
- `jepa-gps-shortcut-benchmark`: 增加真实 perturbation forward 模式、logits cache 输出和 strict evidence scope，避免 deterministic degradation rows 被误用为 primary claim。
- `observability-aware-fusion`: 增加 no-regret gate、anchor fallback 和 reranker branch diagnostics，确保 reliability metadata 不污染普通 baseline。

## Impact

- 受影响模型：`src/kd_sensing/models/` 中新增或扩展 rerank/residual fusion 组件，优先注册为 component；`modular_sequence` 只做 opt-in 接入。
- 受影响评估：`src/kd_sensing/diagnostics/jepa_benchmark_runner.py`、perturbation helpers、evaluation/runtime forward 输出适配和 logits/diagnostics cache 写出。
- 受影响配置：新增 rerank fusion smoke/strict YAML、real-forward benchmark manifest 和 claim gate 配置。
- 受影响数据与 batch：复用现有 difficulty pipeline 和 metadata 字段；必须保证 target label、beam power oracle 和未来信息不进入模型输入。
- 受影响测试：模型 forward、candidate set/rerank loss、condition id isolation、benchmark real-forward metrics、claim gate pending/pass/fail、普通 baseline 忽略 metadata、architecture boundary。
- 运行产物：所有真实 logits cache、CSV/JSON/PNG/checkpoint/TensorBoard 继续写入 ignored `outputs/analysis/real_perturbation_residual_rerank_fusion/`。
