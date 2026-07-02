# 当前模型架构目录

本页用于直接查看当前 `model-architecture` 表面积：有哪些整模型、encoder、projector、representation core 和 head，它们分别适合什么输入、承担什么角色。它是人类可读清单，不是第二套 registry；权威构建来源仍是 `kd_sensing.registries` 和对应 OpenSpec。

快速查看某个配置的真实参数量和组件摘要：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary \
  --config configs/image/supervised.yaml \
  --format markdown
```

查看 JEPA visual sweep 候选的声明参数口径：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.model_architecture_summary \
  --sweep-manifest configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml \
  --variant-id patch14_stage1_gps_query \
  --format csv
```

## 怎么读

- 新增普通 baseline 时，优先使用 `model.primary.type: modular_sequence`，通过 `ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES`、`HEADS` 替换局部组件。
- 只有 paper/workflow reproduction、官方协议包装或不能表达为组件组合的结构，才新增 `MODELS` 整模型例外。
- 参数量以 `model_architecture_summary` 输出为准：实例级模型来自真实 `nn.Module`，sweep 候选来自声明 metadata，并会标记 `parameter_count_source`。
- 下列表格的 `总参数` / `可训练参数` 是典型默认口径；除特别说明外按 `output_dim=d_model=num_classes=64` 估算，多模态 core 以 `K=2` 示例，实际 YAML 可能因模态、层数、冻结策略或 checkpoint 配置不同而变化。
- 图像主线输入默认是 `rgb_imagenet`：`[B, T, 3, 224, 224]`，ImageNet normalization，不在 encoder 内做 resize。

## 推荐主路径

| 使用场景 | 首选模型/组件 | 说明 |
| --- | --- | --- |
| 新增单模态或多模态 supervised baseline | `modular_sequence` | 当前默认组合模型；按模态选择 encoder，再经 projector、representation core 和 beam head 输出 logits。 |
| 图像主线 encoder | `resnet18_imagenet_rgb` | 默认 RGB/ImageNet 图像 encoder；支持冻结 backbone、指定 ResNet stage 解冻。 |
| 图像轻量或规模扫描 | `tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb`、`tinyvit_11m_22k_rgb` | opt-in TinyViT encoder；不替换默认 ResNet-18，不是 KD/distillation workflow。 |
| JEPA image+GPS downstream | `jepa_context_image` | JEPA context visual encoder，可接 mean/GPS-query/hybrid/predictive pooler 和 temporal auxiliary。 |
| GPS / Radar / LiDAR / mmWave / CSI 组件基线 | `gps_mlp`、`radar_cnn`、`lidar_cnn`、`mmwave_mlp`、`pilot_dual_view_csi` | 模块化 encoder，通常输出到 `d_model=64` 后接 core/head。 |
| 快速单帧 baseline | `snapshot_frame` | 要求 `seq_len=1`、`num_pred=1`；适合 next-frame/snapshot 对照。 |
| 时序 baseline | `single_gru`、`early_concat_gru` | 单模态用 `single_gru`，多模态 early concat 用 `early_concat_gru`。 |
| token / query / predictive fusion | `token_transformer`、`next_beam_query_transformer`、`feature_consistency_gate` | 用于 token-aware fusion、下一 beam query 和 JEPA predictive robustness 主线。 |
| GPS geometry prior / safe rerank | `gps_geometry_prior`、`geometry_prior_logit_fusion`、`safe_residual_beam_reranker` | 作为 `modular_sequence` 的 optional heads，不恢复已退役 residual 研究线。 |

## MODELS

| Registry name | 总参数 | 可训练参数 | 类 / 实现 | 角色与当前用途 |
| --- | ---: | ---: | --- | --- |
| `modular_sequence` | 典型 11.243M | 典型 0.066M | `kd_sensing.models.modular.ModularSequenceModel` | 当前主推荐组合模型；可组合 encoder/projector/core/head、geometry prior、logit fusion、safe reranker 和 auxiliary heads。 |
| `resnet18_imagenet_rgb` | 11.209M | 0.033M | `kd_sensing.models.image_encoders.ResNet18ImageEncoder` | 图像 encoder 也可作为整模型注册；通常在 `modular_sequence.encoders.image` 中使用。 |
| `camera_ae_frozen` | 0.802M | 0 | `kd_sensing.models.image_encoders.CameraAEImageEncoder` | BeamBench/Arnold22 Camera AE frozen latent encoder；需要 checkpoint，mock/smoke 可显式关闭要求。 |
| `jepa_context_image` | 0.099M | 0.099M | `kd_sensing.models.jepa.JepaContextImageEncoder` | JEPA visual/context image encoder；支持 visual token metadata、pooler/adapter、GPS-query 和 temporal/predictive metadata。 |
| `gps_conditioned_jepa` | 0.249M | 0.149M | `kd_sensing.models.jepa.GPSConditionedJEPA` | GPS-conditioned JEPA 预训练整模型。 |
| `amr_net` | 配置相关 | 配置相关 | `kd_sensing.models.amr_net.AMRNet` | AMR-Net paper-inspired 三模态概率嵌入与 CUAF 融合 whole-model exception；current architecture baseline，不是旧 source-audit runner。 |
| `pinn_multimodal_beam` | 配置相关 | 配置相关 | `kd_sensing.models.pinn_multimodal_beam.PINNMultimodalBeamModel` | MMW physics-informed whole-model exception；内部耦合 path head、ULA channel synthesizer、physics logits、hybrid logits 和 diagnostics，不能干净拆成普通 head。 |
| `bev_fusion_2604` | 14.768M | 14.768M | `kd_sensing.models.bev_fusion_2604.BEVFusion2604Net` | arXiv:2604.05668 BEV-Fusion 本体复现实验。 |
| `vision_position_late_fusion` | 11.227M | 0.050M | `kd_sensing.models.vision_position.VisionPositionLateFusionNet` | Vision-Position late fusion baseline。 |
| `vision_position_transformer_fusion` | 配置相关 | 配置相关 | `kd_sensing.models.vision_position.VisionPositionTransformerFusionNet` | Vision-Position Transformer fusion baseline。 |
| `gps_sequence_baseline` | 0.034M | 0.034M | `kd_sensing.models.vision_position.GpsSequenceBaselineNet` | GPS-only neural sequence baseline。 |
| `token_transformer_fusion` | 典型 2.981M | 典型 2.981M | `kd_sensing.models.fusion.token_transformer.TokenTransformerFusionNet` | 旧式整模型 token transformer fusion；新扩展优先走 `modular_sequence` core。 |
| `cls_token_transformer_fusion` | 典型 2.990M | 典型 2.990M | `kd_sensing.models.fusion.cls_token_transformer.CLSTokenTransformerFusionNet` | CLS-token fusion baseline，canonical lightweight fusion 可用。 |

## 旧 registry 名称迁移表

下列名称已从 current registry discovery 中移除；只有仍有当前迁移价值的少数名称保留 removed guard，其它低价值历史名回落为普通 unknown-name 诊断。Python 类可继续作为窄导入用于历史 checkpoint 检查或 focused tests，但配置和新实验必须走迁移目标。

JEPA-MSAC 的 `jepa_msac` whole-model exception 已退役删除，不再保留 class/module 导入或 removed wrapper；旧配置应迁移到 GPS-conditioned JEPA、JEPA visual analysis、GPS shortcut benchmark 或 `modular_sequence`。

| 退役名称 | Registry | 迁移目标 |
| --- | --- | --- |
| `modular_sequence_model` | `MODELS` | `modular_sequence` |
| `gps_only_neural_baseline` | `MODELS` | `gps_sequence_baseline` |
| `image_strong`, `image_lightweight` | `MODELS` | `configs/image/{strong,lightweight}.yaml` with `modular_sequence + resnet18_imagenet_rgb + single_gru + beam_head` |
| `radar_strong`, `radar_lightweight` | `MODELS` | `configs/radar/{strong,lightweight}.yaml` with `modular_sequence + radar_cnn + single_gru + beam_head` |
| `gps_strong`, `gps_lightweight` | `MODELS` | `configs/gps/{strong,lightweight}.yaml` with `modular_sequence + gps_mlp + single_gru + beam_head` |
| `lidar_strong`, `lidar_lightweight` | `MODELS` | `configs/lidar/{strong,lightweight}.yaml` with `modular_sequence + lidar_cnn + single_gru + beam_head` |
| `mmwave_strong`, `mmwave_lightweight` | `MODELS` | `configs/mmwave/{strong,lightweight}.yaml` with `modular_sequence + mmwave_mlp + single_gru + beam_head` |
| `fusion_strong`, `fusion_lightweight` | `MODELS` | `modular_sequence` fusion with modality encoders, projectors, `early_concat_gru`, and `beam_head`; lightweight all-modality canonical configs may keep `cls_token_transformer_fusion` |
| `radar_feature_extractor`, `lidar_feature_extractor`, `mmwave_feature_extractor` | `MODELS` | `ENCODERS.radar_cnn`, `ENCODERS.lidar_cnn`, `ENCODERS.mmwave_mlp` or direct class import |
| `point_cloud_mlp` | `ENCODERS` | `lidar_cnn` for current LiDAR BEV configs |
| `jepa_token_transformer` | `REPRESENTATION_CORES` | `token_transformer` or `token_aware_transformer` |
| `safe_residual_reranker` | `HEADS` | `safe_residual_beam_reranker` |

## ENCODERS

| Registry name | 总参数 | 可训练参数 | 模态 / 输入 | 输出与关键 metadata | 备注 |
| --- | ---: | ---: | --- | --- | --- |
| `resnet18_imagenet_rgb` | 11.209M | 0.033M | image `[B, T, 3, 224, 224]` | `[B, T, output_dim]`；metadata 记录 `pretrained`、`weights`、`freeze_backbone`、`trainable_stages`。 | 默认图像主线；`unfreeze_stages` 可选 `conv1/bn1/layer1/layer2/layer3/layer4`。 |
| `tinyvit_5m_scratch_rgb` | 12.103M | 0.021M | image `[B, T, 3, 224, 224]` | `[B, T, output_dim]`；metadata 记录 variant、checkpoint source、freeze policy、backbone/output dim、参数量。 | TinyViT-5M scratch，默认冻结 backbone。 |
| `tinyvit_5m_22k_rgb` | 12.103M | 0.021M | image `[B, T, 3, 224, 224]` | 同 TinyViT；22k checkpoint 会记录 path/url/cache provenance。 | 优先用 `checkpoint_path`；`allow_download=true` 才允许 URL/cache 路径。 |
| `tinyvit_11m_scratch_rgb` | 20.383M | 0.029M | image `[B, T, 3, 224, 224]` | 同 TinyViT。 | TinyViT-11M scratch，规模更大。 |
| `tinyvit_11m_22k_rgb` | 20.383M | 0.029M | image `[B, T, 3, 224, 224]` | 同 TinyViT。 | TinyViT-11M 22k distill checkpoint 变体。 |
| `jepa_context_image` | 0.099M | 0.099M | image `[B, T, 3, 224, 224]`，部分 pooler 需要 GPS context | frame features `[B, T, latent_dim]` 或 token features；metadata 记录 visual token encoder、token count/grid、pooler、adapter、checkpoint policy、temporal auxiliary。 | JEPA downstream 主力 image encoder；`output_dim` 必须等于 `latent_dim`。 |
| `camera_ae_frozen` | 0.802M | 0 | image `[B, T, C, H, W]`，内部可 resize 到 AE `image_size` | `[B, T, output_dim]`；metadata 记录 checkpoint、latent_dim、freeze_encoder。 | BeamBench Camera AE 路线；默认要求 checkpoint。 |
| `gps_mlp` | 0.005M | 0.005M | GPS-Rel-Polar `[B, T, 3]` | `[B, T, output_dim]`；可设 `gps_input_size`、`hidden_size`、`dropout`。 | GPS 组件 baseline 默认 encoder。 |
| `radar_cnn` | 1.151M | 1.151M | radar `[B, T, 2, 128, 64]` | `[B, T, output_dim]`；可设 `radar_channels`/`in_channels`。 | Radar 组件 baseline 默认 encoder。 |
| `lidar_cnn` | 0.126M | 0.126M | LiDAR BEV `[B, T, 3, H, W]` | `[B, T, output_dim]`；可设 `lidar_channels`/`in_channels`。 | LiDAR BEV 默认 encoder，通道通常为 height/intensity/density。 |
| `mmwave_mlp` | 0.017M | 0.017M | mmWave feature `[B, T, mmwave_input_size]` | `[B, T, output_dim]`；可设 `hidden_size`、`dropout`。 | mmWave 组件 baseline 默认 encoder。 |
| `pilot_dual_view_csi` | 0.050M | 0.050M | CSI / pilot dual-view batch | `[B, T, output_dim]` | CSI hardening 与 GPS+CSI 矩阵使用的 encoder。 |

## PROJECTORS

| Registry name | 总参数 | 可训练参数 | 作用 | 输入 / 输出 |
| --- | ---: | ---: | --- | --- |
| `linear` | 0.004M | 0.004M | 将 encoder 输出投影到统一 `d_model`。 | 接收 `[B, T, D]` 或 `[B, T, K, D]`；可选 LayerNorm、dropout，再 Linear 到 `d_model`。 |
| `identity` | 0 | 0 | encoder 输出维度已经等于 `d_model` 时直接透传。 | 要求 `input_dim == d_model`，否则构建时报错。 |

## REPRESENTATION_CORES

| Registry name | 总参数 | 可训练参数 | 输入 | 输出 | 用途 |
| --- | ---: | ---: | --- | --- | --- |
| `single_gru` | 0.025M | 0.025M | 单模态 `[B, T, D]` | `[B, T, hidden_size]` | 单模态时序 baseline。 |
| `early_concat_gru` | 0.037M | 0.037M | 多模态 `[B, K, T, D]` | `[B, T, hidden_size]` | 多模态 early-concat GRU baseline。 |
| `snapshot_frame` | 0.033M | 0.033M | `[B, 1, D]` 或 `[B, K, 1, D]` | `[B, 1, output_dim]` | `seq_len=1` snapshot baseline；不建模长历史。 |
| `token_transformer` | 0.100M | 0.100M | 多模态 token `[B, K, T, D]` | `[B, T, D]` | 标准 token-aware Transformer fusion。 |
| `token_aware_transformer` | 0.100M | 0.100M | 同 `token_transformer` | `[B, T, D]` | `token_transformer` 的语义别名。 |
| `next_beam_query_transformer` | 0.105M | 0.105M | 多模态 `[B, K, T, D]` | `[B, 1, output_dim]` | 加 modality/time embedding 和 next-beam query token，适合 next-beam query 主线。 |
| `feature_consistency_gate` | 0.046M | 0.046M | image/GPS 等多模态 `[B, K, T, D]` | `[B, T, output_dim]` | 用当前图像、历史预测和 GPS residual 生成门控融合；不消费 condition id。 |
| `jepa_feature_consistency_gate` | 0.046M | 0.046M | 同 `feature_consistency_gate` | `[B, T, output_dim]` | JEPA predictive robustness 语义别名。 |

## HEADS

| Registry name | 总参数 | 可训练参数 | 作用 | 备注 |
| --- | ---: | ---: | --- | --- |
| `beam_head` | 0.004M | 0.004M | 标准 beam classification head。 | LayerNorm + Dropout + Linear，输入 `[B, T, D]`，输出 beam logits。 |
| `beam` | 0.004M | 0.004M | `beam_head` 的别名。 | 方便配置写短名。 |
| `gps_geometry_prior` | 0 | 0 | 从 GPS/history 生成 geometry prior logits/distribution。 | 作为 `modular_sequence.geometry_prior` 可选分支启用。 |
| `geometry_prior_logit_fusion` | 0 | 0 | 融合 image logits 与 geometry prior logits。 | 支持 assistive/fusion 语义，记录 branch weight diagnostics。 |
| `safe_residual_beam_reranker` | 231 | 231 | 基于 anchor logits / geometry prior 做安全 rerank。 | 当前 geometry prior 相关 workflow 可用；不恢复退役 GPS residual 或 BGAM 路线。 |

## 摘要字段口径

| 字段 | 含义 |
| --- | --- |
| `total_params` | 模型或组件中去重后的全部参数数量。 |
| `trainable_params` | `requires_grad=True` 的参数数量。 |
| `frozen_params` | `requires_grad=False` 的参数数量。 |
| `effective_params` | 排除明确未参与 downstream forward 的参数后的有效口径。 |
| `excluded_params` / `excluded_parameter_groups` | 被语义排除的参数及原因，例如未使用的分类 head。 |
| `image_encoder_params` | image encoder 角色聚合参数，常用于比较 ResNet/TinyViT/JEPA。 |
| `visual_context_encoder_params` | JEPA visual/context encoder 或 sweep manifest 中的视觉上下文参数口径。 |
| `parameter_count_source` | 参数来源：`actual_module`、`declared_candidate_metadata`、`startup_summary_artifact` 或 mixed。 |

## 退役边界

KD、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS coarse anchor、GPS residual、camera residual、geometry residual、CRAF/MARF/G2D 和 Multimodal-NF 只允许作为历史说明、migration guard 或退役墓碑出现；不要把它们写回当前模型清单、root config 或推荐入口。

## 维护清单

新增或修改模型架构时，同步检查：

- registry：新增组件注册到 `MODELS`、`ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 的正确位置。
- metadata：组件提供或继承 `training_strategy_metadata()`，至少说明 architecture/category、freeze/checkpoint/reliability metadata 和关键维度。
- summary：`model_architecture_summary` 能输出参数量、组件角色、warning 和 source 口径。
- tests：新增 registry build、synthetic forward、metadata、summary 或配置加载 focused tests。
- docs：更新本页；若新增整模型例外，同时更新 OpenSpec、`docs/project_surface_inventory.md` 和必要的主线目录。
