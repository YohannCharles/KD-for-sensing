## Why

现有 Scenario C 已能评估 GPS 异步漂移，但还不足以解释 Image-JEPA 为什么应该在“GPS 不可靠且视觉结构成为唯一稳定线索”时优于 CNN-based fusion。需要新增 Scenario D 图像可观测性坍塌基准，把 GPS 可靠性退化与图像可观测性退化组成可复现、可审计、可画相变图的二维压力测试。

## What Changes

- 新增 Scenario D / Image Observability Collapse Benchmark，用固定 `D0` 到 `D7` 图像可观测性等级评估模型在天气、低光、运动模糊、局部遮挡、帧 dropout、burst missing 和联合 worst-case 条件下的表现。
- 将 Scenario C 的 `C0` 到 `C4` GPS 可靠性等级与 Scenario D 的 `D0` 到 `D7` 图像可观测性等级组成 `performance[Cx, Dy]` 二维鲁棒性矩阵。
- 强制 benchmark 对 GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only、Image-JEPA+GPS 使用相同 split、label space、metric profile 和 corruption seeds。
- 扩展 image difficulty / observability transform，使 image corruption、image missing、valid mask、burst dropout、observability score 和 replay metadata 成为统一 difficulty pipeline 的一部分。
- 新增 observability-aware fusion 能力，使用 `image_observability_score`、`image_valid_mask`、`gps_valid_mask` 和 `gps_delay_steps` 进行 reliability weighting、adaptive fusion 和不确定性 gating。
- 扩展 Image-JEPA downstream，使其在 `C3/C4 + D3/D4/D6/D7` 等优势条件下可显式依赖 temporal latent prediction fallback，而不是退化为 raw CNN feature fusion。
- 新增论文级输出契约：Scenario D CSV、Cx-Dy heatmap、worst-case、RSI、phase transition curves、CNN vs JEPA crossing point 和 modality dominance ratio。
- 不引入 breaking change；现有 JEPA shortcut benchmark、difficulty profile、训练/评估入口和主线文档继续兼容，新增能力通过 manifest/config 显式启用。

## Capabilities

### New Capabilities

- `scenario-d-image-observability-benchmark`: 定义 Scenario D 图像可观测性等级、Cx-Dy 二维鲁棒性矩阵、模型组、指标、输出产物和复现实验边界。
- `observability-aware-fusion`: 定义 image/GPS reliability weighting、adaptive fusion、uncertainty gating、JEPA temporal fallback 触发条件和诊断输出。

### Modified Capabilities

- `modality-difficulty-pipeline`: 增加 image observability transform、`D0` 到 `D7` preset、image missing/dropout 与 physical corruption 区分、observability score 和 replay metadata。
- `jepa-gps-shortcut-benchmark`: 扩展为可运行 Scenario C x Scenario D 二维矩阵，并输出 Scenario D 论文表格、heatmap、phase transition 与 modality dominance 产物。
- `jepa-downstream-extensibility`: 增加 Image-JEPA temporal context encoder / predictive visual fallback 的下游契约，以及 image validity/observability metadata 的模型输入边界。
- `modality-contracts`: 增加 image difficulty metadata 字段语义，至少覆盖 `image_valid_mask`、`image_observability_score`、`image_dropout_mask`、`image_burst_dropout_mask`、corruption type/severity 和 frame range。

## Impact

- 代码影响：新增或扩展 `src/kd_sensing/data/difficulty/` 下的 image observability operator；新增 `src/kd_sensing/models/observability_aware_fusion.py` 或等价窄模块；扩展 JEPA context image encoder / downstream pooler 以消费 image observability metadata 和 temporal history；扩展 benchmark runner、aggregation 和 figure export。
- 配置影响：新增 Scenario D benchmark manifest 或 diagnostic config preset，新增 Cx-Dy matrix、D-level preset、model group 和 output artifact 声明；配置必须继续使用 canonical modality key `image` 和 `gps`，不得新增伪模态名称。
- 入口影响：优先复用现有包内 benchmark / diagnostics CLI；如需新增 CLI，必须是包内 console script，并同步 README、inventory 和架构边界测试；不得新增仓库根旧入口。
- 数据与产物边界：不得改写真实 `dataset/`、split CSV、label、beam target、power target、checkpoint 或训练统计；新增 CSV、NPY、PNG、report、cache 和 runtime manifest 只写入 ignored 的 `outputs/`、`logs/` 或 manifest 指定本地产物目录。
- 测试影响：需要覆盖 transform determinism、shape/dtype/target preservation、no future leakage、score 计算、C/D preset schema、strict comparability、fusion gating、JEPA fallback 触发、metric aggregation 和 CLI/config smoke。
