# Design: Scene31-34 PatchViT Encoder Ablation Fix

## 方案

新增一个薄 encoder wrapper：`lightweight_patchvit_frame`。它不重新实现 ViT，而是复用已有 `JEPA_VISUAL_TOKEN_ENCODERS` 中的 `patch_vit`：

- `Conv2d` patch embedding 作为 1 层 CNN/stem；
- `TransformerEncoderLayer`，默认 `depth=1`；
- 对 patch tokens 做 mean pooling，输出模块化模型需要的 `[B, T, output_dim]`。

该 wrapper 同时接受 `image_channels`、`lidar_channels` 或 `in_channels`，因此 image 和 LiDAR BEV 都能通过同一 registry encoder 表达。这样避免把 ImageNet TinyViT 误当成用户要的轻量 PatchViT。

## 运行拓扑

新增四个 run，使用单独 output root `outputs/scenes31_34_patchvit_lmdb`：

- `scenes31_34_patchvit_image_pretrain_seed1`
- `scenes31_34_patchvit_lidar_pretrain_seed1`
- `scenes31_34_proto_randomdrop_subset_patchvit_es40_seed1`
- `scenes31_34_proto_randomdrop_subset_patchvit_jepa_es40_seed1`

执行仍分两阶段：

1. image/lidar PatchViT 单模态预训练并行。
2. 两个预训练 checkpoint 产出后，并行启动普通 PatchViT downstream 和 PatchViT+JEPA downstream。

现有 TinyViT-5M 运行不关闭、不复用 output root、不覆盖日志。

## 配置原则

- 单模态预训练沿用 M2Beam single-modal Scene31 base，只替换对应模态 encoder。
- Downstream 沿用 Scene31-34 main proto random non-empty subset exposure 设置，只替换 image/lidar encoder 和 checkpoint path。
- PatchViT+JEPA downstream 只启用轻量 JEPA loss 权重，不引入新的结构搜索。

## 验证

- registry/forward focused test：`lightweight_patchvit_frame` 可构建并输出 `[B, T, D]`。
- generator focused test：manifest、run 名、encoder 类型、checkpoint path 和 JEPA flag。
- `openspec validate fix-scenes31-34-patchvit-encoder-ablation --strict`。
- 启动真实训练时使用新的 output root 和 GPU，不影响已有 TinyViT 进程。
