## Why

用户希望引入 radar/LiDAR 后增加真正针对缺失模态鲁棒性的强对照。`Robust Multimodal Beam Prediction With Missing Modality`（IEEE WCL 2025）主题直接对应 missing-modality beam prediction，但当前尚未确认可用官方代码，因此需要单独 change 先做论文/代码审计，再做可审计的 paper-aligned local substitute。

## What Changes

- 新增 WCL 2025 Robust Multimodal Missing-Modality baseline 复现 workflow，先审计论文、官方代码/权重可用性、数据集、模态、split 和 metric 口径。
- 若找到官方代码，则作为 official-code reproduction 包装；若未找到，则实现 paper-aligned local substitute，并明确 claim status。
- 支持 image/LiDAR/radar/GPS 等论文声明模态的缺失模态训练与评估，输出 condition-level metrics 和 strict comparability metadata。
- 将复现状态写入 manifest：`official_reproduction`、`local_substitute`、`blocked`、`not_comparable` 或 `pending`。
- 不把该 baseline 与 AMBER-lite 混淆；AMBER-lite 是本地最小强对照，WCL 2025 是论文对齐复现。

## Capabilities

### New Capabilities

- `wcl2025-robust-missing-modality-reproduction`: 约束 IEEE WCL 2025 缺失模态鲁棒 beam prediction baseline 的 source audit、official/local-substitute 分支、模型/训练复现、评估输出和 claim status。

### Modified Capabilities

- 无。

## Impact

- 代码范围：新增 `src/kd_sensing/baselines/wcl2025_missing_modality/` 或等价窄 workflow owner；必要时新增可组合 missing-modality fusion component。
- 配置范围：新增 paper-aligned reproduction manifest/config，默认输出到 ignored `outputs/analysis/wcl2025_missing_modality_reproduction/`。
- 文档/OpenSpec 范围：新增本 change artifacts；实现时同步主线模型目录、实验协议表和 claim/provenance 账本。
- 验证范围：source-audit manifest tests、official/local-substitute branch tests、synthetic forward、condition summary adapter、OpenSpec validate 和架构边界测试。
