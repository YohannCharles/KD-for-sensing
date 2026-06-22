## Why

当前 P0-P5 heatmap 主要覆盖 Image+GPS、JEPA 和少量非 JEPA clean baseline，缺少公开可复现的 Vision/LiDAR/Radar/GPS 全模态强对照。TII 的 DeepSense6G 2022 challenge 方案使用 camera、LiDAR、radar 和 GPS，并给出代码、预处理、checkpoint 和 Scenario 31-34 DBA，可作为引入 radar/LiDAR 后的第一条外部强 baseline。

## What Changes

- 新增 TII VLRG Transformer baseline 复现 workflow，作为 `workflow/paper reproduction`，不伪装成普通 `modular_sequence` baseline。
- 支持审计 TII 外部源码、预处理脚本、best checkpoint、输入模态、split、DBA 口径和 challenge leaderboard provenance。
- 增加本仓库内的薄 wrapper/adapter，用统一 DeepSense6G scene31-34 协议运行或导入 TII 预测结果，并输出可合并到现有 P0-P5/DBA summary 的 machine-readable rows。
- 明确本地数据、下载的外部 checkpoint、预处理缓存和预测 CSV 只写入 ignored runtime output，不纳入源码。
- 不恢复旧根脚本、旧兼容 facade 或 retired KD/HiST/residual/BGAM 路线。

## Capabilities

### New Capabilities

- `tii-vlrg-transformer-reproduction`: 约束 TII Vision/LiDAR/Radar/GPS Transformer 外部强 baseline 的源码审计、数据适配、运行入口、provenance、指标输出和产物边界。

### Modified Capabilities

- 无。

## Impact

- 代码范围：新增 `src/kd_sensing/baselines/tii_vlrg_transformer/` 或等价窄 workflow owner、包内 CLI、必要的 summary adapter；可能复用 `src/kd_sensing/data/datasets/deepsense6g*`、`src/kd_sensing/evaluation` 和 diagnostics summary helper。
- 配置范围：新增 baseline reproduction 配置或 manifest，默认指向 ignored `outputs/analysis/tii_vlrg_transformer_reproduction/`，外部 repo/checkpoint 路径通过用户配置传入。
- 文档/OpenSpec 范围：新增本 change artifacts；实现时同步主线模型目录、实验协议表和 claim/provenance 账本。
- 验证范围：OpenSpec strict validate、CLI help smoke、manifest/provenance unit tests、synthetic prediction-summary adapter tests；真实 DeepSense6G 数据和外部 checkpoint 运行作为 opt-in，不作为单元测试依赖。
