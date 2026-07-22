## Why

现有 inner-only missing-evidence probe 表明 full-minus-missing residual 对 image 与 LiDAR 缺失具有恢复潜力，但尚未在统一的静态基础模型、真实 bypass 语义和 topology-aware objective 下比较 learned residual、train-mean residual、共享映射与非线性容量。需要一次 single-seed、claim-ineligible 快速验证，判断该方向是否值得继续，而不重训或解冻四模态主模型。

## What Changes

- 审计 B2 standalone-quality 的静态化条件；若无合规 train-fit global/static prior，则按预注册规则固定退回 C0 corruption augmentation + learned global prior，并让 R0-R5 共用同一基础模型、split 与缓存。
- 新增只含四模态 prototype beam evidence、full/missing logits、residual target 和分层 metadata 的 inner-train/inner-validation residual cache；拒绝 outer test 与 channel/path/gain/power 字段。
- 新增 full、missing radar/GPS 和非目标 mask 严格 bypass 的轻量 residual wrapper，以及 R1 train-mean、R2/R3 modality-specific linear、R4 shared mask-conditioned linear、R5 modality-specific MLP 六组快速验证。
- 使用 train-only normalization、loss 量级校准、固定 batch order、inner-validation total loss checkpoint 选择和现有 cyclic beam topology，统一输出恢复比例、天气/sector/错误距离、动态替换、效率与 success gates。
- 新增 GPU0-5 独立失败隔离 launcher；完成本轮后停止，不运行 outer test、multi-seed 或下一轮训练。

## Capabilities

### New Capabilities

- `beam-topology-missing-residual-adapter`: 定义冻结四模态静态基础模型上的 missing-image/LiDAR residual cache、bypass adapter、六组快速验证、诊断和停止边界。

### Modified Capabilities

- `training-evaluation-runtime`: 允许从审计过的 inner clean cache 训练 validation-loss-selected 的 claim-ineligible residual adapter，并约束静态基础模型、checkpoint 选择和运行终止边界。

## Impact

变更集中在 `analysis/` 的独立缓存/runner、`scripts/` 的 GPU0-5 launcher、针对性测试和 OpenSpec；所有 checkpoint、CSV、日志与报告写入 ignored `outputs/missing_residual_adapter/`。不新增依赖，不修改 prototype bank、canonical config、默认模型 forward 或正式 claim。
