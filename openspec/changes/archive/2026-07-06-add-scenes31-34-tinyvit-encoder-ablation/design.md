# Design: Scene31-34 TinyViT Encoder Ablation

## 方案

采用最小实现：不新增 registry 或模型代码，直接通过配置把主模型的 `encoders.image` 和 `encoders.lidar` 都切换为现有 `tinyvit_5m_scratch_rgb`。因为当前 Scene31-34 主 workflow 的 image 和 lidar BEV 输入均为 3-channel 224x224 tensor，该 encoder 可以作为公平的轻量 ViT-style encoding 对照。

## 运行拓扑

四个 run：

- `scenes31_34_tinyvit_image_pretrain_seed1`
- `scenes31_34_tinyvit_lidar_pretrain_seed1`
- `scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed1`
- `scenes31_34_proto_randomdrop_subset_tinyvit_jepa_es40_seed1`

执行时分两阶段：

1. pretrain stage：image/lidar 单模态 TinyViT encoder 预训练并行。
2. downstream stage：读取两个 pretrain checkpoint 后，非 JEPA 与 JEPA downstream 并行。

这满足用户要的 4 个程序总量和阶段内并行，同时避免 downstream 在 checkpoint 未产出时失败。

## 配置原则

- 单模态预训练沿用 M2Beam single-modal 基础协议，只替换 encoder 并把 dataset 扩展到 Scene31-34 pooled split。
- Downstream 沿用 Scene31-34 main proto random non-empty subset exposure 设置，只替换 image/lidar encoder 和 checkpoint path。
- JEPA downstream 只打开 `loss.u_mask_beam_jepa.enabled/use_jepa_loss` 与小权重 `lambda_jepa`，避免把对照组变成新一轮大规模结构搜索。

## 验证

- 生成脚本测试：manifest 数量、run 名、TinyViT encoder 类型、checkpoint path、random subset 和 JEPA flag。
- `openspec validate add-scenes31-34-tinyvit-encoder-ablation --strict`。
- 启动真实训练前 runner 写出 scene availability 和 worker status；训练产物保留在 `outputs/`。
