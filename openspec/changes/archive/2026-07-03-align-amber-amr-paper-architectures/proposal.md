## Why

当前 AMBER full 与 AMR-Net 都已作为本地 baseline 接入，但实现仍带有若干 paper-inspired 简化，容易在论文复现口径中被误解为严格架构对齐。用户已明确指定 AMBER 与 AMR-Net 的论文对齐修订范围，需要在保持当前训练入口和 scene31 实验边界的前提下收敛模型结构。

## What Changes

- AMBER full 去掉历史 beam index 路径与 `learned_history_beam_token`，只保留 image、radar、LiDAR、GPS 四个模态输入。
- AMBER full 按论文补齐空间 token/位置编码与 Class-Former-aided Modality Alignment 语义，避免继续只用每模态全局向量和简化 CMA loss 表达完整论文结构。
- AMBER image、radar、LiDAR encoder 统一使用 ResNet-18 风格路径，并在配置中开启预训练权重；不再使用 AMBER 专属 ResNet34 image caveat。
- AMR-Net 按论文修订输入与训练目标：RGB/灰度论文输入口径、K 次 Monte Carlo PRE、论文版 CUAF entropy/KL/top-T margin 公式、训练损失不额外强加 fused focal 主损失。
- AMR-Net 保持当前 scene31 本地实验配置，不切换到论文 Scenario 8/9。
- AMBER full 与 AMR-Net 的本地默认输入长度统一调整为 `seq_len=2`，并允许模型 forward 消费不同输入时间长度；AMR-Net 在进入 snapshot encoder 前对时间维做 mean pooling。
- 保持两条 baseline 的 current 入口、claim 边界和本地产物输出规则，不恢复退役 `amr_net_gps_image` runner。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `amber-full-architecture-reproduction`: AMBER full 的 architecture contract 改为四模态输入、无历史 beam token、ResNet18 pretrained encoder、空间 token/位置编码和论文式 CMA。
- `amr-net-architecture`: AMR-Net 的 architecture/loss contract 改为论文式输入通道、K 次 PRE、论文版 CUAF、AMR-only composite training objective 和可变输入时间长度，同时保留 scene31 配置边界。

## Impact

- 影响 `src/kd_sensing/models/amber_full.py`、AMBER 相关 encoder/config/loss/test。
- 影响 `src/kd_sensing/models/amr_net.py`、`src/kd_sensing/losses/amr_net.py`、`configs/fusion/amr_net_supervised.yaml` 和 focused tests。
- 影响模型架构 metadata、claim caveat 和 focused validation；不改变通用训练入口、数据集 loader 或退役入口策略。
