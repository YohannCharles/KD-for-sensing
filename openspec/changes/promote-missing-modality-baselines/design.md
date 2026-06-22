## Context

现有三条缺失模态相关路线来自论文复现 change：AMBER-lite 已基本是本地可训练 baseline，WCL2025 有 local substitute 配置但训练期 dropout 字段没有接入通用 difficulty pipeline，TII 只有 external workflow wrapper。用户当前目标不是复现论文，而是在自己的 DeepSense6G/MMW 实验场景里跑可训练 baseline。

约束：

- 所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。
- 不依赖官方源码、开源权重、外部 checkpoint 或下载权重。
- 不新增训练循环；baseline 统一走 `kd-sensing-train` 和现有 `modular_sequence`。
- 输出仍写入 ignored `outputs/` 范围，不提交本地数据、cache、log 或 checkpoint。

## Goals / Non-Goals

**Goals:**

- AMBER-lite 默认无需下载权重即可训练和验证。
- WCL-style baseline 的训练期缺失模态扰动通过 `difficulty.profiles` 生效。
- TII-VLRG-style baseline 提供本仓库可训练配置，而不是只依赖 external wrapper。
- 文档和 claim 账本把三者表述为 local experimental baselines。

**Non-Goals:**

- 不删除已有 source-audit / external import wrapper。
- 不声称 AMBER、WCL2025 或 TII official reproduction。
- 不新增外部依赖或复制论文官方训练代码。
- 不做完整长训或数值 claim。

## Decisions

1. 训练入口统一使用 `kd-sensing-train --config ...`。
   - 这样复用现有 dataloader、batch contract、loss、evaluation 和 artifact writer。
   - TII local baseline 用 `modular_sequence` + all-modalities Transformer，而不是 external wrapper。

2. 默认不使用外部权重。
   - AMBER-lite image encoder 改为 `pretrained: false`、`weights: null`。
   - WCL-style 和 TII-style local configs 也保持 scratch/local training。

3. 缺失模态训练只走 difficulty pipeline。
   - WCL-style baseline 将 `training.modality_dropout` 迁移为 `difficulty.profiles`，避免配置字段存在但训练不消费。
   - AMBER-lite 继续使用已接入的 `amber_lite_modality_dropout` operator。

4. 保留旧 wrapper 的边界。
   - `kd-sensing-tii-vlrg-transformer` 和 `kd-sensing-wcl2025-missing-modality-audit` 仍可用于 external/audit。
   - 文档不再把它们列为用户实验 baseline 的主路径。

## Risks / Trade-offs

- [Risk] TII-style local baseline 不是 TII 官方模型。→ Mitigation: metadata 和文档标记 `local_experimental_baseline`，不进入 official claim。
- [Risk] 默认 scratch ResNet 可能弱于 ImageNet 初始化。→ Mitigation: 默认满足“不用开源权重”；需要预训练时用户可显式 override。
- [Risk] WCL-style 缺失模态训练和 AMBER-lite 相近。→ Mitigation: WCL-style 保留五模态 token transformer；AMBER-lite 保留四模态 mask-token transformer。
