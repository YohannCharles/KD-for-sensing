## Context

当前仓库的普通 baseline 首选 `modular_sequence`，复杂论文模型只有在 OpenSpec 明确理由时才允许新增完整 `MODELS` 注册名。用户提供的 AMR-Net 方案包含三模态独立概率嵌入、per-modality classifier、训练期 FEP/PRE loss 以及推理期 CUAF logit/probability 融合，输出中还需要保留 `mu/logvar`、采样 latent、模态 logits 和 uncertainty diagnostics。

现有 `ModelOutput` 适配可以消费 dict 输出：`logits` 作为主 beam logits，其它字段进入 diagnostics。现有 batch/runtime 已能向声明 `supports_modality_kwargs=True` 的模型传入 `image_batch`、`lidar_batch`、`gps_batch`，因此 AMR-Net 不需要新增训练循环或 batch 分支。历史 `AMR-Net_gps_image` runner 和配置 token 已退役，本 change 只新增新的 current 能力，不复活旧入口。

## Goals / Non-Goals

**Goals:**

- 实现 `amr_net` whole-model exception，支持 image、LiDAR、GPS 三模态输入和 beam 分类输出。
- 提供概率嵌入、FEP/PRE loss helper、CUAF inference fusion 和可审计 diagnostics。
- 保持现有训练、评估、配置加载、registry、ModelOutput 和本地产物边界。
- 增加 focused tests，覆盖 registry build、synthetic forward、loss、CUAF、metadata、架构摘要和旧入口隔离。

**Non-Goals:**

- 不恢复 `amr_net_gps_image`、`kd-sensing-run-amr-net-gps-image` 或任何根目录训练脚本。
- 不新增独立 `dataset.py`、`train.py`、`eval.py` 小工程，也不复制通用训练循环。
- 不承诺论文数值复现、天气/SNR/dropout 完整实验矩阵或官方源码一致性；本 change 只交付可训练架构和最小验证入口。
- 不改变全局 batch contract；论文中的 `[B, ...]` 输入映射为仓库的 `[B, T, ...]`，AMR-Net 初版要求 `T=1`。

## Decisions

### 1. 使用 whole-model exception，而不是强塞进 `modular_sequence`

AMR-Net 的主训练信号来自每个模态的概率分布和分类器，而主推理输出来自 CUAF 对 per-modality probabilities 的动态融合。把这些拆成普通 encoder/core/head 会迫使训练侧理解多个 head 的 KL/PRE/FEP 细节，反而扩大共享 runtime。新增 `MODELS.register("amr_net")` 更小，且符合现有 whole-model exception 契约。

备选方案是新增 `amr_probabilistic_core` 和 `cuaf_head` 组件。该方案以后可做，但初版需要改动 `modular_sequence` 对多 head loss diagnostics 的假设，范围更大。

### 2. 模型输出使用 dict，主 `logits` 为 CUAF fused logits/probability logits

`forward()` 返回 dict，至少包含 `logits`、`modality_logits`、`mu`、`logvar`、`z`、`cuaf_weights`、`uncertainty` 和 `metadata`。训练模式默认仍计算 CUAF fused logits 以兼容现有 beam loss/metrics，但 AMR loss helper 使用 diagnostics 中的 per-modality 字段计算 FEP/PRE。推理时 `logits` 必须代表 CUAF 融合结果。

备选方案是训练时只返回某个模态 logits，评估时返回 fused logits。该方案会让同一模型 train/eval 输出语义变化过大，不利于测试和摘要。

### 3. 输入保持仓库 batch 形状，初版只支持 snapshot `T=1`

AMR-Net 论文描述的是单帧输入。实现接受 `image_batch [B,T,C,H,W]`、`lidar_batch [B,T,N,F]` 或可配置 flattened LiDAR、`gps_batch [B,T,F]`，并在初版要求 `T=1`。这避免修改 dataset 与 runtime；后续如果要支持历史窗口，可新增 temporal pooling，而不是现在预留复杂接口。

### 4. Image/LiDAR/GPS encoder 直接放在 AMR 模型文件内的私有子模块

这些 encoder 只服务 AMR-Net 的概率嵌入尺寸和 per-modality classifier，不作为通用组件暴露。Image encoder 使用轻量 ResNet-style stem/blocks，支持配置 `image_channels=1|3`；LiDAR encoder 使用 `Conv1d + MLP`；GPS encoder 使用 MLP。这样不新增通用 registry surface。

备选方案是把三个 encoder 都注册到 `ENCODERS`。初版没有其它模型复用它们，注册会增加公共面。

### 5. FEP/PRE 做成一个窄 loss helper

新增 `amr_net_loss_from_output(output, labels, cfg)` 或等价 objective helper，读取 `ModelOutput.diagnostics` 中的 `amr` 字段，计算 per-modality CE、KL 和监督式 contrastive PRE。KL 默认 `alpha=0.01`，PRE 默认 `K=2`、`temperature=0.1`，并允许配置关闭 PRE 用于 smoke。

备选方案是把 loss 直接写进模型 forward。这样会把标签传进模型，破坏现有训练/评估分层。

### 6. CUAF 是模型内部 inference/fusion helper，不消费 benchmark condition id

CUAF 使用 entropy、cross-modal KL consistency 和 top-k margin 计算模态权重，并输出 diagnostics。它只消费 logits/probabilities 和可选模态 availability mask，不读取 `condition`、`c_idx`、`d_idx` 等报告字段。

## Risks / Trade-offs

- [Risk] PRE 在小 batch 或单类 batch 上没有正样本，可能产生 NaN → loss helper 必须跳过无正样本 anchor，并记录 skipped count；全空时 PRE 返回 0。
- [Risk] LiDAR 真实 batch 形状可能不是论文的 `[216,2]` → 实现用配置声明 `lidar_input_features` 和 shape 校验，focused test 覆盖 paper-shape；真实配置需要按 dataset profile 调整。
- [Risk] ResNet-style image encoder 与论文 lightweight ResNet 不完全一致 → metadata 标记 `paper_approximation=true` 和 encoder 配置；不声明 official reproduction。
- [Risk] 新 whole-model 注册名扩大 public surface → 只新增 `amr_net` 一个 current 名称，旧 `amr_net_gps_image` 仍由 migration guard/unknown-name 路径拒绝。
- [Risk] CUAF fused probability 转 logits 时数值不稳定 → 使用 `log(p.clamp_min(eps))` 作为 `logits`，并在测试中覆盖 finite 输出。

## Migration Plan

1. 新增 AMR-Net 模型、loss helper、配置和默认组件导入。
2. 添加 synthetic focused tests，并运行 OpenSpec 与相关 pytest。
3. 更新最小文档目录条目，说明这是 current AMR-Net architecture baseline，不是旧 source-audit runner。
4. 回滚时删除 `amr_net` 注册、配置、loss helper、tests 和文档条目；旧退役入口不受影响。

## Open Questions

- 真实 DeepSense6G LiDAR 输入最终采用 raw rays `[216,2]`、BEV cache，还是当前 dataset 现有 LiDAR tensor 形状；实现前以当前 dataset contract 为准。
- AMR-Net 是否需要列入主线实验矩阵的正式 run，还是只作为可选 paper-inspired baseline；本 change 初版只要求架构可训练和 smoke。
