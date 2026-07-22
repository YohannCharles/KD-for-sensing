## Why

现有 PR-SQDF clean cache 已保存冻结 C0 的四模态 block evidence、global prior 和 full fused logits，但尚不清楚剩余三模态能否恢复完全缺失模态的 evidence，或直接恢复 full-minus-missing residual。需要一次 single-seed、inner-only、claim-ineligible 的小型离线 probe，以最终 beam 指标和 oracle-gap recovery 判断是否值得开发 fallback adapter。

## What Changes

- 新增只读取 PR-SQDF clean cache 的 missing-evidence probe，严格排除 outer test、channel、CSI、path、gain、beam power 和原始传感器张量。
- 按 C0 固定时间 prior 精确聚合 modality-level evidence，构造四种 missing logits、evidence target、full-minus-missing residual 和 oracle。
- 固定比较 mean、train-only nearest neighbor、linear/MLP evidence、linear/MLP residual 与 oracle；四个缺失模态共享 split、seed、batch order、normalization、optimizer 和 validation recovery-objective checkpoint 规则。
- 输出缓存重建 gate、predictability/recovery 指标、weather/beam-sector 分层、效率审计、16 个任务状态和唯一可行性建议；结果仅写入 ignored `outputs/missing_evidence_probe/`。
- 增加四 GPU launcher；单任务只负责一个缺失模态，结束后不自动启动 multi-seed、outer test 或完整 fallback 训练。

## Capabilities

### New Capabilities

- `missing-evidence-predictability-probe`: 定义冻结 clean cache 上的 modality evidence/residual 目标、无泄漏轻量 probe、重建 gate、统一指标和可行性停止边界。

### Modified Capabilities

- `training-evaluation-runtime`: 允许从审计过的 frozen clean cache 训练 claim-ineligible 小型 recovery probe，并限定 inner-validation recovery objective 的 checkpoint 选择与禁止 outer-test 的边界。

## Impact

变更涉及 `analysis/` 的独立 cached probe、`scripts/` 的 GPU0--3 launcher、针对性测试和 OpenSpec。不会修改 C0 checkpoint、prototype bank、canonical config、默认模型 forward 或正式 claim；不会新增第三方依赖，所有运行产物保留在 ignored `outputs/`。
