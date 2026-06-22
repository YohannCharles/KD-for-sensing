## Why

当前 heatmap 已有 Image+GPS、GPS-only、CNN/TinyViT/JEPA 对照，但缺少训练时显式处理 image/LiDAR/radar/GPS 缺失和退化的鲁棒 baseline。AMBER-lite 以最小结构复现 AMBER 类缺失模态思想，提供可控、可训练、可解释的 full-multimodal robustness 对照。

## What Changes

- 新增 AMBER-lite missing-modality baseline：每个模态独立 encoder，缺失模态使用 mask token，融合 core 使用轻量 Transformer 或 token-aware core。
- 训练配置支持 modality dropout，对 image、LiDAR、radar 和 GPS 进行可配置随机缺失，记录实际 dropout policy。
- 评估配置支持 clean、missing-modality、poor image、LiDAR/radar unavailable、wrong/async GPS 和组合扰动摘要。
- 输出统一 DBA/P0-P5 或 condition-level summary row，并记录 reliability/missing-mask metadata、strict comparability 和 provenance。
- 不声称完整 AMBER 官方复现；该 change 只实现 AMBER-lite 本地强 baseline。

## Capabilities

### New Capabilities

- `amber-lite-missing-modality-reproduction`: 约束 AMBER-lite 缺失模态 baseline 的模型结构、modality dropout 训练、missing-mask metadata、评估输出和产物边界。

### Modified Capabilities

- 无。

## Impact

- 代码范围：新增或复用 `ENCODERS`、`REPRESENTATION_CORES`、difficulty/modality mask helper、配置 preset 和 summary adapter；优先使用 `modular_sequence`，只有无法表达时才新增窄 component。
- 配置范围：新增 AMBER-lite train/eval YAML 或 manifest，默认输出到 ignored `outputs/analysis/amber_lite_missing_modality/`。
- 文档/OpenSpec 范围：新增本 change artifacts；实现时同步主线模型目录、实验协议表和 result claim 账本，明确 `lite/local reproduction` 状态。
- 验证范围：模型 registry/config load、synthetic forward、missing-mask/dropout policy、condition-level summary adapter 和 OpenSpec validate。
