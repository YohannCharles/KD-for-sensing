## Why

现有 JEPA visual architecture sweep 显示 `patch14_stage1_gps_query`、`overlap_k16_s8_stage1` 和 `resnet18_layer4_tokens` 处在非常接近的性能区间，但它们混合了视觉分辨率、CNN 局部归纳偏置、JEPA Stage 1 checkpoint、ImageNet 预训练、冻结策略和 checkpoint selection 等变量。下一步需要一次覆盖 CNN、CNN+Transformer hybrid、JEPA 预训练和蒸馏路线的完整实验矩阵，用同一协议把“局部先验是否比继续提高 patch 分辨率更重要”验证清楚。

## What Changes

- 扩展 JEPA visual architecture sweep，新增全量实验族：CNN token anchors、ImageNet-pretrained CNN variants、CNN-token JEPA Stage 1、CNN+ViT/hybrid tokenizers、patch/overlap 邻域补点、ResNet teacher 到 patch/hybrid student 蒸馏、冻结/微调策略、pooler/core ablation 和 compute/parameter controls。
- 新增可生成完整实验矩阵的配置生成器、job manifest、并行运行脚本和汇总脚本；默认输出位于 ignored `outputs/analysis/cnn_hybrid_jepa_visual_prior_sweep/`。
- 为每个候选记录严格可比性 metadata：variant family、token source、token count、checkpoint policy、pretraining source、freeze policy、teacher/distillation source、parameter count、compute proxy、checkpoint selection policy、split/metric provenance。
- 支持两阶段及三阶段运行：可选 Stage 1 JEPA pretraining、supervised downstream、best/best_top1 checkpoint re-evaluation；蒸馏候选增加 teacher 训练或 teacher checkpoint 复用阶段。
- 不删除、不重命名现有 `fair_gps_biased`、`gps_query_pool`、`patch14_stage1_gps_query`、`overlap_k16_s8_stage1` 或 `resnet18_layer4_tokens` 结果路径；本 change 只新增更完整的实验计划和可复现实验脚手架。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `jepa-visual-architecture-sweep`: 扩展 visual architecture sweep 的候选矩阵、运行脚本和汇总契约，使其覆盖 CNN 局部先验、CNN+Transformer hybrid、JEPA Stage 1 复用、预训练/冻结策略和 CNN teacher distillation 的全量对照。

## Impact

- 影响配置与实验脚手架：`configs/pretraining/`、`configs/fusion/experiments/jepa_image_gps/`、`configs/diagnostics/` 或生成到 `outputs/analysis/.../generated_configs/` 的 YAML。
- 影响模型组件时仅通过现有 registry 边界新增 opt-in visual token encoder、adapter、pooler 或 distillation loss，不恢复旧 KD 路线，不新增绕过 `src/kd_sensing` 包结构的长期入口。
- 影响运行产物：训练日志、checkpoints、metrics、CSV/JSON/Markdown summary 全部写入 ignored runtime output 目录。
- 验证包括 `openspec validate add-cnn-hybrid-jepa-visual-prior-sweep --strict`、生成配置加载 smoke、相关模型 forward smoke、架构边界测试和最终 sweep summary 完整性检查。
