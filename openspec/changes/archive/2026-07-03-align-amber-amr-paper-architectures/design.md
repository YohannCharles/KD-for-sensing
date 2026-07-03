## Context

AMBER full 当前以 `modular_sequence` representation core 形式实现，已具备缺失模态 mask、fusion token、modality/fusion transformer 和 AMBER auxiliary loss，但仍把历史 beam 表述为 learned token，并且默认 encoder 多数输出每模态每时间步一个全局向量。AMR-Net 当前作为 `amr_net` whole-model exception 实现，具备概率嵌入、per-modality classifier、PRE/FEP 辅助 loss 和 CUAF，但 PRE/CUAF/训练目标仍是本地近似。

用户已明确要求：AMBER 去掉 beam index 与 learned history token，仅保留 image/radar/LiDAR/GPS；AMBER 的空间 token 和 CMA 按论文修；image/radar/LiDAR 使用 ResNet18 并开启预训练。AMR-Net 的 PRE、CUAF、训练目标和输入通道按论文修；实验 scene 保持 scene31。

## Goals / Non-Goals

**Goals:**

- 将 AMBER full 的 core 改成四模态论文对齐路径，不再构造历史 beam token。
- 让 AMBER image/radar/LiDAR encoder 能输出 ResNet18 feature-map token，core 能消费 `[B,K,T,S,D]` 或等价展平 token，并使用空间/时间/模态位置编码。
- 将 AMBER CMA 改为 class-query cross-attention 风格 payload，并让 auxiliary loss 消费 class query 对齐 logits。
- 将 AMR-Net 的 PRE 改为 K 次 Monte Carlo latent sampling，CUAF 改为论文 entropy、average pairwise KL、top-T margin 和分项 softmax 归一化。
- 将 AMR-Net 默认训练目标切到 AMR composite loss，避免额外 fused focal 主损失污染论文目标。
- 让 AMBER full 和 AMR-Net 都可消费不同输入时间长度，并将本地默认输入长度改为 `seq_len=2` 以匹配主线模型。

**Non-Goals:**

- 不切换 AMR-Net 到论文 Scenario 8/9；scene31 继续作为当前本地实验场景。
- 不恢复退役 `amr_net_gps_image` runner、旧 mock/source-audit 工作流或专用训练循环。
- 不承诺 AMBER/AMR-Net official 数值复现；仍保持 local/pending claim 边界。
- 不引入新深度学习依赖；优先使用已有 torch/torchvision 路径。
- 不为 AMR-Net 新增时序 Transformer 或 RNN；可变输入长度通过进入 snapshot encoder 前的轻量时间聚合实现。

## Decisions

1. AMBER 去掉 `include_history_beam` 和 `history_beam_token` 行为。
   - 采用：core token 序列只包含 fusion token 与四个模态 token。
   - 备选：保留 learned history token 但默认关闭。不采用，因为用户明确要求去掉，且保留配置会继续制造论文口径歧义。

2. AMBER 空间 token 使用 ResNet18 feature map 的最小改造路径。
   - 采用：新增或扩展 ResNet18-backed encoder，使 image/radar/LiDAR 可以输出每帧 feature-map tokens；没有 token 输出的模态保留单 token。
   - 备选：为每个模态重写完整论文 encoder。不采用，改动面过大且不必要；当前目标是架构语义对齐，不是官方训练代码复刻。

3. AMBER CMA 用 class-query cross-attention 模块替代当前简化 logits。
   - 采用：core 内维护 per-modality class queries 和 fusion query，训练时输出 query embeddings 与 contrastive logits；loss helper 只消费该 payload。
   - 备选：继续用 fusion/modality cosine logits。不采用，因为用户要求第 3 点按论文修改。

4. AMR-Net PRE 在 loss helper 中重新采样。
   - 采用：loss helper 使用 `mu/logvar` 和 `pre_samples` 生成 K 次 latent，再按 supervised contrastive 公式计算 PRE。
   - 备选：forward 直接返回所有 K samples。不采用，训练外 diagnostics 更重；loss helper 重采样是最小可测实现。

5. AMR-Net 训练目标通过配置禁用额外 beam 主损失。
   - 采用：新增配置开关，让 `compute_prediction_loss` 在 AMR-only 模式下不叠加 fused focal 主损失，只记录 AMR composite loss。
   - 备选：写 AMR 专用训练循环。不采用，违反现有共享 runtime 边界。

6. 可变输入长度采用最小架构改造。
   - 采用：AMBER full 继续使用现有 time embedding 与 transformer 消费运行时 `T <= max_seq_len`；AMR-Net 将 `[B,T,...]` 沿时间维 mean pooling 成 snapshot 后进入原三模态 encoder。
   - 备选：为 AMR-Net 添加专用 temporal encoder。不采用，因为这会把 snapshot baseline 改成新的时序模型，超出用户要求。

## Risks / Trade-offs

- [Risk] ResNet18 pretrained 权重在离线环境不可下载。→ Mitigation: 使用 torchvision weights API 时遵循现有 `weights`/`pretrained` 配置；测试不下载权重，真实训练由本地环境决定是否已缓存。
- [Risk] 空间 token 会增加显存。→ Mitigation: 配置保留可控 token pooling/flatten 上限，默认只对 AMBER full 启用。
- [Risk] AMR-only loss 改变通用训练损失组合。→ Mitigation: 只由 `loss.amr.paper_objective_only=true` opt-in，普通 baseline 保持原语义。
- [Risk] 论文式 CUAF 在极端缺失模态时分母不稳定。→ Mitigation: 保留 eps、availability fallback 和 finite focused tests。
- [Risk] AMR-Net mean pooling 会弱化时间顺序信息。→ Mitigation: 该模型仍是 snapshot baseline；focused tests 覆盖不同 `T` 的形状与 mask 语义，若后续需要时序建模再引入显式 temporal encoder。
