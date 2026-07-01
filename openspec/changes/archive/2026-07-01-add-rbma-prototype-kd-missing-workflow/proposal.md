## Why

现有 U-MaskBeamJEPA 和 AMBER full 已经具备 missing mask、reliability-gated fusion、Gaussian JEPA loss 和本地 AMBER-style 复现基础，但缺少一套面向缺失模态 beam prediction 的第二阶段增强：hard mask + soft reliability attention、beam topology 对齐、full-to-partial teacher stabilization、pattern-balanced missing schedule 和可横向比较的 ablation 矩阵。

本 change 用于把用户给出的五阶段实验提示词收敛为可实施、可验证、可回滚的 OpenSpec 方案，优先服务 `no_jepa` full-modality teacher 到 partial-modality student 的鲁棒训练，同时保持现有训练入口、数据契约和退役 KD guard 不被破坏。

## What Changes

- 新增 `ReliabilityBiasedMissingAwareAttention` 作为 U-MaskBeamJEPA 的 opt-in fusion 类型，覆盖 hard missing key/value mask、soft reliability log-bias、learnable beam query、可选 global token 和 attention diagnostics。
- 新增 beam-aware prototype alignment 能力，包含 beam prototype bank、Gaussian/circular soft beam target、fused/modality/teacher feature prototype loss、可选 supervised contrastive loss，以及 prototype top-k diagnostics。
- 新增 full-to-partial teacher stabilization，先实现 online full teacher：同一模型使用 full mask 产生 stop-gradient teacher logits/features，再用 sampled missing mask 训练 student；checkpoint teacher 只保留配置和接口占位，不作为首轮实现承诺。
- 新增 pattern-balanced missing mask sampler，用配置显式提高 `missing_gps`、`non_gps_only`、`only_gps` 和随机高缺失率样本覆盖，并支持按 pattern 聚合训练/评估指标。
- 新增 no-JEPA 优先的 ablation 配置矩阵，覆盖 AMBER-style hard mask baseline、RBMA、RBMA+prototype、RBMA+KD、RBMA+prototype+KD 和小权重 JEPA 对照。
- 更新 U-MaskBeamJEPA、soft beam label、实验 workflow、模型扩展和主线实验文档契约，明确这些增强是 opt-in current/local workflow，不恢复 legacy KD、旧 distillation runtime、旧根脚本或独立训练框架。

## Capabilities

### New Capabilities

- `rbma-prototype-kd-missing-workflow`: 覆盖 RBMA fusion、beam prototype alignment、full-to-partial teacher stabilization、pattern-balanced missing sampler、ablation 配置和按 pattern 评估的完整增强 workflow。

### Modified Capabilities

- `u-mask-beam-jepa`: 增加 `reliability_biased_missing_attention` fusion、no-JEPA KD/prototype 训练选项、pattern-balanced mask 和新增 diagnostics 契约。
- `soft-beam-label-training`: 复用 beam topology soft target 语义，新增 prototype alignment 中的 beam-neighborhood target 要求，并保持其非 KD 命名边界。
- `experiment-workflow`: 增加缺失 pattern ablation 配置与评估入口要求；当前训练仍使用配置驱动入口，不新增根目录训练/评估框架。
- `model-architecture-extension-contract`: 明确 RBMA、prototype bank 和 KD/prototype losses 的扩展路径分别落在 opt-in model/loss/helper 边界，普通 baseline 不被新增 metadata 污染。
- `mainline-experiment-documentation`: 要求文档账本记录 RBMA/prototype/KD ablation 的 pending/local status、比较口径、运行命令和 claim caveat。

## Impact

- 代码：主要影响 `src/kd_sensing/models/u_mask_beam_jepa.py`、新的窄模型/attention helper、`src/kd_sensing/losses/`、missing mask helper、trainer loss wiring、eval pattern helper、配置和 focused tests。
- API：新增 opt-in config 字段；不改变普通 baseline forward 必填输入，不要求非 U-MaskBeamJEPA 模型消费 reliability、prototype 或 KD fields。
- 数据与产物：测试使用 synthetic tensor；训练/评估输出仍写入 ignored `outputs/`，不读取或提交真实 `dataset/`、checkpoint、cache 或日志。
- 依赖：不新增第三方依赖；实现使用 PyTorch、现有 registry/config/runtime、当前 missing mask 和 evaluation 边界。
- 风险：该 change 会重新引入“teacher guidance/KD”命名空间，必须明确它是 U-MaskBeamJEPA 内部 opt-in full-to-partial stabilization，不恢复 retired radar KD、legacy distillation runtime 或 teacher checkpoint 默认入口。
