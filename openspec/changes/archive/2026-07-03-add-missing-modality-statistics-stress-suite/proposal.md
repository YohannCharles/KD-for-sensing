## Why

缺失模态论文不能只靠单 seed clean accuracy 或少数固定缺失 pattern。当前 U-MaskBeamJEPA/Scene31 workflow 已经能产出 pattern-level 指标，下一步需要补统计显著性、真实 stress-curve 和 claim gate，让结果从“看起来更好”升级为“可写进论文的鲁棒性证据”。

## What Changes

- 新增 missing-modality statistical evidence 能力，支持按 method/seed/pattern 汇总 mean、std、bootstrap CI、paired delta、win/loss count 和显著性检验。
- 扩展 U-MaskBeamJEPA eval matrix，输出 strict comparability fields、pattern group summary、seed aggregation 输入和可供 claim harvester 消费的稳定 schema。
- 扩展缺失模态 stress suite，覆盖 clean、single missing、multi missing、only GPS、non-GPS、random missing severity、image degradation、GPS noise/async、radar/LiDAR unavailable 等条件。
- 复用 `modality-difficulty-pipeline`，所有 stress 条件只扰动输入和 mask/reliability metadata，不移动 target、beam power、sample id 或 split metadata。
- 为 Scene31 next-round / RBMA / AMBER-lite/full / RMBP-MM 等本地 baseline 定义 claim-oriented benchmark manifest 口径。
- 不替代 predictive JEPA 专属 stress suite；本 change 聚焦缺失模态鲁棒性，predictive JEPA 只作为可复用 difficulty/stress 设计参考。

## Capabilities

### New Capabilities

- `missing-modality-statistical-evidence`: 覆盖多 seed 统计汇总、bootstrap/permutation 或 Wilcoxon-style paired test、effect size、claim gate 和表格输出 schema。
- `missing-modality-stress-suite`: 覆盖缺失模态真实 stress-curve manifest、condition taxonomy、severity sweep、strict comparability 和输出产物边界。

### Modified Capabilities

- `u-mask-beam-jepa-eval-matrix`: 增加 seed aggregation、strict comparability fields、pattern group summary 和 stress suite 输出要求。
- `modality-difficulty-pipeline`: 增加缺失模态 stress suite 所需的 image/GPS/radar/LiDAR/mmWave unavailable/noise/async 条件标准化要求。
- `local-missing-modality-baselines`: 增加 AMBER-lite/full、RMBP-MM、U-MaskBeamJEPA 等本地 baseline 的 stress benchmark comparability 要求。
- `mainline-experiment-documentation`: 增加统计显著性和 stress suite claim 升级要求。

## Impact

- 主要影响 eval matrix、difficulty profile、diagnostics/summary 脚本、Scene31/RBMA summary、claim registry 文档和 focused tests。
- 输出 CSV/JSON/Markdown/图表默认位于 ignored `outputs/analysis/missing_modality_stress/` 或显式本地输出目录。
- 测试使用 synthetic/mock metrics 和小 fixture，不读取真实 `dataset/`，不提交真实 metrics、figures、checkpoint 或 cache。
