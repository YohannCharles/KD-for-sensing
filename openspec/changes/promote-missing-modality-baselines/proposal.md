## Why

当前 AMBER-lite、WCL2025 local substitute 和 TII VLRG 路线混有“论文复现 / official artifact / external wrapper”的语义，但用户目标是把这些结构当作本仓库实验场景中的可训练 baseline。需要把默认入口收敛到不依赖开源权重、可直接用本地数据训练和验证的 local baseline。

## What Changes

- 将 AMBER-lite 默认配置改为本地 baseline：不默认下载 ImageNet 权重，保留训练期 missing-modality dropout 和 mask-token fusion。
- 将 WCL2025 local substitute 改为本地 baseline：不再把 official reproduction/audit 作为训练主入口，训练期缺失模态扰动接入现有 `difficulty.profiles`。
- 为 TII VLRG Transformer 增加本仓库可训练的 VLRG-style Transformer baseline 配置；外部 wrapper 仅保留为可选 external import/audit，不作为默认 baseline。
- 更新主线模型目录、实验协议、claim 账本和实验矩阵，使这些条目明确为 local experimental baselines，而不是论文复现。
- 保持外部源码、checkpoint、cache、metrics 和训练产物仍写入 ignored runtime output，不提交本地数据或权重。

## Capabilities

### New Capabilities

- `local-missing-modality-baselines`: 约束 AMBER-lite、WCL-style 和 TII-VLRG-style baseline 在本仓库实验场景中的本地训练、验证、配置、metadata 和 claim 边界。

### Modified Capabilities

- 无。

## Impact

- 代码范围：最小配置和测试改动；必要时只新增/复用 `modular_sequence` baseline 配置，不新增训练循环。
- 配置范围：`configs/fusion/amber_lite_missing_modality.yaml`、`configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml`、新增 TII local trainable config。
- 文档范围：`docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md` 和 README 简短索引。
- 验证范围：focused baseline tests、config load、synthetic/real smoke 训练路径、OpenSpec strict validate 和架构边界测试。
