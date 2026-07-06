# Proposal: Scene31-34 TinyViT Encoder Ablation

## 背景

Scene31-34 主实验当前主模型使用 ResNet18 image encoder 和轻量 lidar CNN，并复用单模态预训练 checkpoint。用户需要一个更直接的 encoder 对照组：把主模型的 image/lidar encoding 都换成已有轻量 TinyViT 风格 encoder，先分别预训练 image 与 lidar encoder，再按主实验协议训练 downstream；同时增加一组启用 JEPA loss 的 TinyViT 预训练 encoder downstream 版本。

## 变更

- 新增 local/manual 生成脚本，产出 Scene31-34 TinyViT encoder ablation 的 4 个配置与 manifest。
- 新增 local/manual runner，按依赖分两阶段执行：先并行跑 image/lidar TinyViT 单模态预训练，再并行跑非 JEPA 与 JEPA downstream 对照。
- 保持产物边界：generated configs、logs、checkpoints、fresh eval 仍写入 ignored `outputs/` root，不进入源码。

## 非目标

- 不新增新的模型结构实现；复用已注册的 `tinyvit_5m_scratch_rgb`。
- 不替换当前主方法，也不把该对照组纳入 final ranking。
- 不提交训练输出、checkpoint、cache 或日志。

## 风险与缓解

- TinyViT 用于 lidar BEV 是同一 3-channel 224x224 encoder 的配置复用，语义上是 encoder 对照而非 lidar 专用新结构；manifest 和配置标签会明确记录。
- Downstream 依赖两份预训练 checkpoint，因此 runner 不会在 checkpoint 缺失时启动 downstream，而是先完成 pretrain stage。
