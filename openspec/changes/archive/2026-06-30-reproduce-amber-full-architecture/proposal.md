## Why

现有 AMBER-lite baseline 只覆盖缺失模态的本地简化融合，尚未复现论文 AMBER 的 adaptive multimodal mask transformer、CMA class-query 对齐和训练期辅助损失。该 change 用于把用户给出的论文架构落到当前模块化模型体系中，形成可训练、可测试、claim 边界清晰的完整架构复现计划。

## What Changes

- 新增 paper-aligned AMBER full architecture reproduction，覆盖 image、LiDAR、radar、GPS 和历史 beam embedding 的输入嵌入、位置编码、可学习 fusion token、缺失模态可感知 attention mask、modality-specific transformer、modality-fusion transformer、CMA/Class-Former 和 beam prediction head。
- 新增训练期 AMBER auxiliary outputs 与 loss 接入要求，包括 L2 reconstruction/embedding alignment、CMA contrastive loss 和 beam focal loss 的加权总损失；推理期只依赖 fusion token 到 beam head 的路径。
- 扩展 AMBER-lite 与 local missing-modality baseline 边界，明确 lite 与 full reproduction 的配置、metadata、claim status 和输出目录区分。
- 新增 synthetic forward、mask attention、loss wiring、metadata、配置加载、架构摘要和文档账本任务；不读取真实 `dataset/`，不提交 checkpoint、metrics、cache 或训练输出。
- 不新增旧式根脚本、兼容聚合层或完整训练循环；默认复用 `kd-sensing-train --config`、`modular_sequence` 组件注册和现有 difficulty/evaluation 边界。

## Capabilities

### New Capabilities

- `amber-full-architecture-reproduction`: AMBER 论文架构的本地完整复现能力，覆盖模块化模型组件、缺失模态 mask、CMA auxiliary loss、metadata、配置、测试和 claim 边界。

### Modified Capabilities

- `amber-lite-missing-modality-reproduction`: 区分 AMBER-lite 与 AMBER full reproduction 的 scope、metadata、输出目录和 claim 边界。
- `local-missing-modality-baselines`: 将 AMBER full reproduction 纳入本地缺失模态 baseline 家族，并保持默认训练入口、扰动 profile 和 claim guard。
- `model-architecture-extension-contract`: 明确 AMBER full architecture 作为 component baseline 优先实现；若后续必须 whole-model exception，需在 design/tasks 中给出不可组合理由和额外测试。
- `mainline-experiment-documentation`: 要求主线模型目录、实验协议表和 claim 账本记录 AMBER full reproduction 的 pending/local status 与 caveat。

## Impact

- 代码：主要影响 `src/kd_sensing/models/modular.py` 或窄模型组件模块、`src/kd_sensing/losses/` 或 objective helper、`configs/fusion/`、`configs/diagnostics/`、`tests/` 和当前实验文档。
- API：新增或扩展 representation core、可选 auxiliary loss/metadata 字段和配置；不改变普通 baseline forward 必填输入，不新增训练 CLI。
- 数据与产物：默认输出写入 ignored `outputs/analysis/local_baselines/amber_full_architecture/`；测试使用 synthetic tensor 或 dry-run manifest。
- 依赖：不新增第三方依赖，优先使用 PyTorch、现有 registry、difficulty pipeline 和 evaluation/summary helper。
