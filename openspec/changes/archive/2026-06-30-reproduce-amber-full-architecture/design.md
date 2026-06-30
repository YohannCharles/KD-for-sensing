## Context

论文 arXiv:2512.11331v2 提出的 AMBER 架构包含五类输入：image、LiDAR、radar、GPS 和历史 beam index。当前仓库已有 `amber_lite_missing_modality_transformer`，能用 mask token 处理 image/radar/GPS/LiDAR 缺失，但它没有论文中的 adaptive multimodal mask attention、modality-specific transformer、learnable fusion token、CMA/Class-Former 和 contrastive auxiliary loss。

仓库当前模型扩展契约要求普通 baseline 优先走 `modular_sequence` 的 encoder/projector/core/head 组件路径；paper reproduction 或 whole-model exception 需要 OpenSpec 明确理由。AMBER 的核心差异集中在 fusion core、auxiliary outputs 和 loss wiring，仍可复用现有 DeepSense6G dataset、difficulty pipeline、`kd-sensing-train --config`、`ModelOutput` 适配和 local baseline claim guard。

## Goals / Non-Goals

**Goals:**

- 提供 paper-aligned AMBER full architecture 的本地可训练复现配置。
- 复用现有 image/LiDAR/radar/GPS encoder/projector，新增历史 beam embedding 和 AMBER fusion core。
- 在训练期输出 modality-specific branch、fusion token、CMA logits/embeddings 和 loss metadata；推理期保留 fusion token beam prediction 路径。
- 让普通 baseline 不需要 AMBER 专用 metadata 或 loss 字段。
- 用 synthetic tests 覆盖 forward、缺失 mask、auxiliary loss、metadata 和配置加载。

**Non-Goals:**

- 不承诺官方 AMBER 数值、Table 复现或排行榜 claim。
- 不自动下载论文源码、外部 checkpoint 或真实数据。
- 不新增旧式根脚本、专用训练循环、兼容聚合层或新的 dataset 解析路径。
- 不把 AMBER-lite 重命名为完整 AMBER；两者继续保留不同 scope。

## Decisions

1. AMBER full 默认作为 `modular_sequence` representation core 扩展，而不是新的完整 `MODELS.register(...)`。
   - 理由：论文结构的输入 encoder、投影、fusion 和 beam head 可以映射到现有模块化边界；新增整模型会扩大 registry surface。
   - 备选：新增 whole-model exception。只有当 `modular_sequence` 无法表达训练期 auxiliary outputs 或 loss metadata 时才启用，并需补 registry build、forward/output adaptation 和架构摘要测试。

2. 新增 `amber_full_adaptive_mask_transformer` core，复用现有 AMBER-lite mask metadata 入口。
   - core 接收 `[B,K,T,D]` 多模态特征和 availability mask，内部加入 modality/time embedding、learnable fusion token、缺失模态可感知 attention mask、modality-specific transformer block 和 modality-fusion transformer block。
   - core 输出主 `[B,1,D_out]` 或 `[B,T,D_out]` fusion representation，并通过 diagnostics/auxiliary 字段暴露训练期分支结果。
   - 备选：直接扩展 `amber_lite_missing_modality_transformer`。不采用，因为 lite 的 mean-fusion 语义应保持稳定，避免改变已有 baseline。

3. 历史 beam index 用窄 encoder/projector 或 core 内 embedding 表达。
   - 优先实现为 `ENCODERS`/projector 中的 `beam_history_embedding`，输入来自现有 batch 中历史 beam 字段；若现有 batch 字段不可用，则任务先补最小 batch contract 测试。
   - 备选：在 dataset 中新增 AMBER 专用字段。不采用，除非现有 target/history contract 完全无法复用。

4. CMA 和 contrastive loss 作为 opt-in auxiliary loss，不污染普通 beam loss。
   - AMBER 配置显式开启 `loss.auxiliary.amber_cma_contrastive`、`loss.auxiliary.amber_l2` 或等价字段。
   - 训练端只在模型输出包含对应 auxiliary payload 时计算，缺失 payload 时对 AMBER 配置早失败，对普通配置忽略。
   - 备选：把 contrastive loss 写进模型 forward。不采用，loss 属于训练配置和 objective 边界。

5. Claim 和文档状态保持 local experimental / pending。
   - 配置、metadata、文档和 claim registry 使用 `reproduction_scope: amber_full_local`。
   - 真实指标、checkpoint、cache、figures 和 reports 只写入 ignored `outputs/analysis/local_baselines/amber_full_architecture/`。

## Risks / Trade-offs

- [Risk] 论文图与公式未给出所有 hidden size、temperature、loss weight 细节 → 用配置显式记录默认值，任务中保留可调字段，claim 不升级为 official reproduction。
- [Risk] 训练期 auxiliary payload 可能与现有 `ModelOutput` 适配不兼容 → 先用 synthetic forward 和 `adapt_model_output` focused test 固定输出字段。
- [Risk] 历史 beam 输入字段在不同 dataset/config 下不可用 → 默认配置只启用已有 DeepSense6G sequence contract，缺字段时早失败并说明需要历史 beam。
- [Risk] AMBER full core 变大后 `models/modular.py` 继续膨胀 → 若实现超过窄改范围，拆到 `src/kd_sensing/models/amber.py` 并只在默认组件导入中注册。
- [Risk] 缺失模态 mask 只 zero-fill 但 attention 仍看见缺失 token → attention mask 测试必须验证 fusion token 不能 attend 到 unavailable modality tokens。
