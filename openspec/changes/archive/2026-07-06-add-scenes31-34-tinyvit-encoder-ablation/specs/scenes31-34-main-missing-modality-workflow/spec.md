## ADDED Requirements

### Requirement: Scene31-34 TinyViT encoder 对照组
项目 MUST 提供 local/manual workflow，用于运行 Scene31-34 主模型的 TinyViT encoder 对照组。该 workflow MUST 生成两个单模态预训练配置和两个 downstream 配置：一个普通 prototype + random subset exposure downstream，一个加入 JEPA loss 的 downstream。所有训练、fresh eval、日志和 checkpoint MUST 写入 ignored runtime artifact root，默认不得纳入源码变更。

#### Scenario: TinyViT ablation 配置生成
- **WHEN** 用户运行 TinyViT encoder ablation config generator
- **THEN** manifest MUST 包含 `scenes31_34_tinyvit_image_pretrain_seed1`、`scenes31_34_tinyvit_lidar_pretrain_seed1`、`scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed1` 和 `scenes31_34_proto_randomdrop_subset_tinyvit_jepa_es40_seed1`
- **AND** image/lidar 预训练配置 MUST 使用 `tinyvit_5m_scratch_rgb`
- **AND** downstream 配置 MUST 把 `model.primary.encoders.image.type` 和 `model.primary.encoders.lidar.type` 设置为 `tinyvit_5m_scratch_rgb`
- **AND** downstream 配置 MUST 通过 `encoder_checkpoint_paths.image` 和 `encoder_checkpoint_paths.lidar` 指向同一 output root 下的预训练 checkpoint

#### Scenario: TinyViT ablation runner 依赖顺序
- **WHEN** 用户运行 TinyViT encoder ablation runner
- **THEN** runner MUST 先并行启动 image/lidar TinyViT 单模态预训练
- **AND** runner MUST 在两个预训练 checkpoint 可用后才启动 downstream jobs
- **AND** downstream stage MUST 并行启动普通 TinyViT downstream 与 TinyViT+JEPA downstream
- **AND** runner MUST 为每个 worker 设置 `CUDA_VISIBLE_DEVICES`，并写出 completed、skipped、failed 和 missing_checkpoint 列表

#### Scenario: TinyViT+JEPA 标记清晰
- **WHEN** generator 写出 TinyViT+JEPA downstream 配置
- **THEN** `model.primary.use_jepa_loss` MUST 为 true
- **AND** `loss.u_mask_beam_jepa.enabled` 和 `loss.u_mask_beam_jepa.use_jepa_loss` MUST 为 true
- **AND** 非 JEPA TinyViT downstream MUST 保持 `loss.u_mask_beam_jepa.enabled=false`
