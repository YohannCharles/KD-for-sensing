## Context

仓库已经有两条与 `paper/` 目录对应且由本 change 完成的本地 baseline：

- AMBER full：已有 `amber_full_adaptive_mask_transformer` core 和 `configs/fusion/amber_full_architecture.yaml`，但 image encoder 仍使用 ResNet18 spatial-token 配置，且未显式输出论文中的 modality-weight L2 regularization payload。
- AMR-Net：已有 `amr_net` whole-model exception、AMR loss helper 和 focused tests，配置已经使用 `seq_len=2`、`num_pred=1`，但协议文档仍写成 snapshot `1 / 1`。

RMBP-MM / WCL 2025 的 source-audit、official artifacts 和完整 local substitute surface 并未由本 change 完成，因此从本 change 的完成声明中移除。H5/P1 当前使用的 `rmbp_mm` channel-attention component 作为真实 consumer 保留，但不等价于 WCL official/local-substitute workflow 完成。

本轮用户要求强约束：输入窗口必须为 2，预测窗口必须为 1，输入模态只能使用 `lidar`、`image`、`gps`、`radar`。论文中不满足该约束的 historical beam / beam measurement / mmWave 或更长窗口字段必须排除，不能作为模型输入恢复。

## Goals / Non-Goals

**Goals:**

- 将三篇论文 baseline 的可运行本地入口统一到 `seq_len=2`、`num_pred=1`。
- 将所有本地训练配置限制为 `image/radar/gps/lidar` 或其论文必要子集；AMR-Net 使用论文三模态 `image/lidar/gps`。
- 补齐 AMBER image ResNet34 spatial-token encoder 与 modality indicator L2 payload。
- 保持所有论文 baseline 的 claim 状态为 local reproduction / local substitute / pending，不声明官方数值复现。

**Non-Goals:**

- 不恢复 historical beam index、beam measurement、mmWave、CSI、旧 AMR-Net_gps_image runner、旧根目录训练脚本或兼容 facade。
- 不下载官方源码、权重或真实数据；不提交 `dataset/`、`outputs/`、`logs/`、cache 或 checkpoint。
- 不把 RMBP-MM/WCL 写成本 change 已完成内容，也不删除其它 current workflow 正在消费的 `rmbp_mm` component。

## Decisions

1. **AMBER image encoder 新增 ResNet34 spatial-token 组件。**  
   论文 image branch 使用 ResNet34，radar/LiDAR 使用 ResNet18。相比把 AMBER 做成整模型例外，新增 `resnet34_spatial_tokens` encoder 更符合当前 component-baseline 边界，也能复用 `modular_sequence`、projector/core/head 和架构摘要。

2. **AMBER 不恢复 historical beam token。**  
   论文 AMBER 使用 `image/lidar/radar/gps/beam index` 五类 token 和滑窗 `W=5`，但用户明确只能使用四个 sensing/GPS 模态且输入窗口为 2。因此配置和 metadata 必须继续声明 `history_beam_usage=disabled`，只在允许的四模态上复现 transformer/mask/CMA/L2 机制。

3. **AMR-Net 保持三模态 whole-model exception。**
   论文 AMR-Net 本身只使用 RGB/Image、LiDAR 和 GPS；这三者都在用户允许集合内。因此不新增 radar 分支，以免把非论文结构混进 AMR-Net。

## Risks / Trade-offs

- [Risk] AMBER 论文使用 ResNet34、beam token 和 `W=5`，本地配置只能使用四模态与 `seq_len=2`。→ Mitigation：metadata 和 docs 明确记录用户约束下的 paper-aligned local architecture reproduction，不声明 official reproduction。
- [Risk] 新 ResNet34 spatial-token encoder 依赖 torchvision 权重枚举，环境或版本可能缺失。→ Mitigation：沿用 ResNet18 helper 风格，测试中关闭 pretrained/weights，真实训练才使用本地环境可用权重。
- [Risk] AMBER core 消费 missing-modality metadata 后影响普通 baseline。→ Mitigation：仅显式支持 missing-modality metadata 的 core 走 mask-aware 调用，普通 modular_sequence baseline 保持原 forward 契约。
