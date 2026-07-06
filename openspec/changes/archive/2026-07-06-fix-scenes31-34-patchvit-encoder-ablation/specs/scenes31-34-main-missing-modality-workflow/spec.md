## ADDED Requirements

### Requirement: Scene31-34 PatchViT encoder 对照组
项目 MUST 提供 local/manual workflow，用于运行 Scene31-34 主模型的轻量 PatchViT encoder 对照组。该 workflow MUST 与 TinyViT-5M 对照组使用不同 run name 和 output root，并 MUST 不停止或覆盖现有 TinyViT 长跑。

#### Scenario: PatchViT ablation 配置生成
- **WHEN** 用户运行 PatchViT encoder ablation config generator
- **THEN** manifest MUST 包含 `scenes31_34_patchvit_image_pretrain_seed1`、`scenes31_34_patchvit_lidar_pretrain_seed1`、`scenes31_34_proto_randomdrop_subset_patchvit_es40_seed1` 和 `scenes31_34_proto_randomdrop_subset_patchvit_jepa_es40_seed1`
- **AND** image/lidar 预训练配置 MUST 使用 `lightweight_patchvit_frame`
- **AND** downstream 配置 MUST 把 `model.primary.encoders.image.type` 和 `model.primary.encoders.lidar.type` 设置为 `lightweight_patchvit_frame`
- **AND** downstream 配置 MUST 通过 `encoder_checkpoint_paths.image` 和 `encoder_checkpoint_paths.lidar` 指向同一 PatchViT output root 下的预训练 checkpoint

#### Scenario: PatchViT ablation 两阶段并行执行
- **WHEN** 用户运行 PatchViT encoder ablation 长跑
- **THEN** image/lidar PatchViT 单模态预训练 MUST 并行执行
- **AND** 两个预训练 checkpoint 可用后，普通 PatchViT downstream 与 PatchViT+JEPA downstream MUST 并行执行
- **AND** 所有 runtime outputs、logs、generated configs 和 checkpoints MUST 写入 ignored output/log roots

#### Scenario: PatchViT+JEPA 标记清晰
- **WHEN** generator 写出 PatchViT+JEPA downstream 配置
- **THEN** `model.primary.use_jepa_loss` MUST 为 true
- **AND** `loss.u_mask_beam_jepa.enabled` 和 `loss.u_mask_beam_jepa.use_jepa_loss` MUST 为 true
- **AND** 非 JEPA PatchViT downstream MUST 保持 `loss.u_mask_beam_jepa.enabled=false`
