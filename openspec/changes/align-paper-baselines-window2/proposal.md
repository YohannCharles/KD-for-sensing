## Why

`paper/` 目录中的 AMBER、AMR-Net 和 RMBP-MM 三篇论文 baseline 当前存在复现口径不一致：AMR-Net 与 AMBER full 已有本地架构复现骨架，但文档、配置和部分组件仍与用户指定的 `seq_len=2`、`num_pred=1`、仅 `lidar/image/gps/radar` 模态边界不完全一致；RMBP-MM 仍是 token-transformer local substitute，未实现论文核心的随机可用模态 masking、相似样本补模态和 channel-attention 融合。

本 change 将三篇论文 baseline 收敛为可审计的本地复现基线：严格遵守本轮输入/预测窗口和模态限制，同时保留无法声明官方数值复现的 claim 边界。

## What Changes

- 将 AMBER full 默认协议固定为 `seq_len=2`、`num_pred=1`，四模态仅 `image/radar/gps/lidar`，并补齐论文对齐的 image ResNet34 spatial-token encoder、AMBER modality indicator / L2 regularization metadata。
- 保持 AMR-Net 为 `image/lidar/gps` 论文三模态 whole-model exception，但明确它仍落在允许模态集合内，并修正文档协议为 `seq_len=2`、`num_pred=1`。
- 将 RMBP-MM local substitute 改为 paper-aligned component baseline：四模态 `image/radar/gps/lidar`、`seq_len=2`、`num_pred=1`，新增 RMBP channel-attention fusion core，并提供随机可用模态 masking 与相似样本补模态的本地 batch augmentation helper。
- 更新 configs、tests、OpenSpec delta 和研究协议文档，确保缺失模态复现仍标记为 local substitute / local experimental baseline，不升级为 official reproduction。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `amber-full-architecture-reproduction`: 调整 AMBER full 的论文对齐 encoder、窗口、模态和 L2 regularization 要求。
- `amr-net-architecture`: 明确 AMR-Net 在用户指定窗口和允许模态集合下的本地复现边界。
- `wcl2025-robust-missing-modality-reproduction`: 将 RMBP-MM local substitute 从 generic token transformer 改为论文对齐的 channel-attention fusion 与 missing-modality augmentation。
- `local-missing-modality-baselines`: 统一本地缺失模态 baseline 的 `seq_len=2`、`num_pred=1` 和 `image/radar/gps/lidar` 模态边界。

## Impact

- 影响模型组件：`src/kd_sensing/models/` 中 image spatial-token encoder 与 RMBP-MM representation core。
- 影响复现 workflow：`src/kd_sensing/baselines/rmbp_mm/` source-audit / local-substitute metadata 与 batch augmentation helper。
- 影响配置：AMBER full、AMR-Net、RMBP-MM local substitute YAML。
- 影响测试：AMBER full、AMR-Net、WCL/RMBP-MM focused tests，以及必要的 registry/config focused checks。
- 不新增旧 CLI、根目录训练脚本、外部下载依赖、真实数据产物、checkpoint 或输出日志到源码。
