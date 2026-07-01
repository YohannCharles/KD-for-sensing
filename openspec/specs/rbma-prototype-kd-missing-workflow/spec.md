# rbma-prototype-kd-missing-workflow Specification

## Purpose
定义 U-MaskBeamJEPA 的 RBMA prototype/KD missing-modality workflow，包括 reliability-biased missing-aware attention、beam prototype alignment、full-to-partial teacher stabilization、pattern-balanced mask sampling、ablation config 和数值安全边界。
## Requirements
### Requirement: Reliability-biased missing-aware attention
系统 MUST 提供 `ReliabilityBiasedMissingAwareAttention` 模块，用于从 canonical 模态 latent 中聚合 beam-relevant fused feature。该模块 MUST 接收 `z: [B, M, D]`、`mask: [B, M]`、`reliability: [B, M]` 或 `[B, M, 1]`、可选 `global_token: [B, D]` 和可选 `global_reliability: [B, 1]`，并 MUST 输出 `fused: [B, D]`、attention weights 和 debug diagnostics。

#### Scenario: 缺失模态不能作为 key/value
- **WHEN** `mask[:, i]` 为 false 或 0
- **THEN** 第 `i` 个模态 MUST 在 attention logits 中被 hard mask
- **AND** softmax 后该模态 attention weight MUST 在数值容差内等于 0

#### Scenario: reliability bias 降低低可靠模态权重
- **WHEN** 两个可用模态 latent 相同但 reliability 不同
- **THEN** attention logits MUST 加入 `beta_reliability * log(reliability + eps)` 或等价单调 bias
- **AND** 低 reliability 模态的 attention weight MUST 不高于高 reliability 模态

#### Scenario: global token 处理全缺失样本
- **WHEN** 某个样本所有传感器模态缺失且未提供 `global_token`
- **THEN** 模块 MUST 抛出清晰错误
- **AND** 系统 MUST 不产生 NaN fused feature

#### Scenario: global token 可作为兜底 token
- **WHEN** 某个样本所有传感器模态缺失但提供了 `global_token`
- **THEN** 模块 MUST 允许 forward 成功
- **AND** attention diagnostics MUST 包含 global token 的有效权重

### Requirement: Beam prototype alignment
系统 MUST 提供 opt-in beam prototype alignment。该能力 MUST 包含 `BeamPrototypeBank`、beam-neighborhood soft target 生成、fused feature prototype loss、可用单模态 prototype loss、可选 teacher feature prototype loss 和可选 supervised contrastive loss。

#### Scenario: prototype bank 输出 beam logits
- **WHEN** `BeamPrototypeBank` 收到 feature `[B, D]`
- **THEN** 系统 MUST 对 feature 和 prototypes 做 normalize
- **AND** 输出 MUST 为 shape `[B, num_beams]` 的 cosine-similarity logits

#### Scenario: soft beam target 归一化
- **WHEN** `make_soft_beam_labels(labels, num_beams, sigma, circular)` 被调用
- **THEN** 返回 target MUST 为 `[B, num_beams]`
- **AND** 每个样本 target 概率和 MUST 在数值容差内等于 1
- **AND** circular 模式启用时 beam 0 与最后一个 beam MUST 按环形距离相邻

#### Scenario: modality prototype loss 只使用可用模态
- **WHEN** 提供 `modality_features: [B, M, D]` 和 `mask: [B, M]`
- **THEN** prototype modality loss MUST 只对 `mask==1` 的模态计算
- **AND** 缺失模态 MUST 不贡献 prototype gradient 或 sample count

#### Scenario: supervised contrastive 无正样本时跳过
- **WHEN** batch 中某个 anchor 没有同 label 正样本
- **THEN** supervised contrastive loss MUST 跳过该 anchor
- **AND** loss MUST 保持有限且 backward 可运行

### Requirement: Online full-to-partial teacher stabilization
系统 MUST 支持 opt-in online full-to-partial teacher stabilization。启用时，同一模型 MUST 先使用 full mask 产生 stop-gradient teacher logits/features，再使用 sampled missing mask 产生 student logits/features，并将 logit KD、feature KD 和可选 prototype KD 加入总损失。

#### Scenario: teacher 输出不更新 teacher 分支
- **WHEN** `training.use_full_to_partial_kd=true` 且 `kd_teacher_mode=online_full`
- **THEN** teacher logits 和 teacher feature MUST detach 或在 no-grad 上下文中产生
- **AND** student loss backward MUST 不通过 teacher 输出更新 teacher 分支计算图

#### Scenario: no-JEPA 模式可使用 KD
- **WHEN** `use_jepa_loss=false` 且 full-to-partial KD 启用
- **THEN** loss MUST 不访问必需 JEPA-only `mu_B` 或 `logvar_B` 字段
- **AND** student beam CE、logit KD 和 feature KD MUST 能完成 backward

#### Scenario: checkpoint teacher 首轮不静默启用
- **WHEN** 配置 `kd_teacher_mode=checkpoint`
- **THEN** 系统 MUST 要么明确实现 checkpoint teacher 加载、eval 和冻结语义
- **AND** 要么抛出清晰 pending 错误
- **AND** 系统 MUST NOT 静默回退为错误的 online 或 legacy teacher runtime

### Requirement: Pattern-balanced missing mask sampler
系统 MUST 提供 pattern-balanced missing mask sampler，用于训练和评估中显式控制缺失模态模式分布。该 sampler MUST 使用 canonical 模态顺序，返回 mask、pattern names 和可选 pattern ids，并 MUST 不原地修改 batch。

#### Scenario: pattern 概率采样
- **WHEN** 调用 `sample_pattern_balanced_mask(batch_size, modalities, pattern_probs, ensure_at_least_one=True)`
- **THEN** 返回 mask MUST 为 `[B, M]`
- **AND** `pattern_names` 长度 MUST 等于 batch size
- **AND** 采样 1000 个样本时各 pattern 比例 MUST 大致符合归一化后的配置概率

#### Scenario: 固定 pattern mask
- **WHEN** pattern 为 `full`、`missing_gps`、`non_gps_only` 或 `only_gps`
- **THEN** mask MUST 分别等价于 `[1,1,1,1]`、`[1,1,1,0]`、`[1,1,1,0]` 和 `[0,0,0,1]`
- **AND** `missing_gps` 和 `non_gps_only` MUST 在日志中保留不同 pattern name

#### Scenario: random pattern 至少一个可用
- **WHEN** pattern 为 `random_0.5`、`random_0.75`、`missing_one_random` 或 `only_one_random`
- **THEN** 每个样本 MUST 至少有一个可用模态
- **AND** mask dtype MUST 能被 U-MaskBeamJEPA forward 安全转换为 boolean availability mask

### Requirement: Ablation configuration matrix
系统 MUST 提供一组 opt-in ablation 配置，用于比较 no-JEPA RBMA、prototype alignment、full-to-partial KD 和 AMBER-style hard mask baseline。配置 MUST 使用当前训练入口和 canonical config 结构。

#### Scenario: 主候选配置启用 RBMA prototype KD
- **WHEN** 用户加载主候选 no-JEPA RBMA prototype KD 配置
- **THEN** 配置 MUST 设置 `use_jepa_loss=false`
- **AND** 配置 MUST 启用 `fusion_type: reliability_biased_missing_attention`
- **AND** 配置 MUST 启用 `use_beam_prototype_alignment=true` 和 `use_full_to_partial_kd=true`

#### Scenario: AMBER-style hard mask baseline 不使用 reliability bias
- **WHEN** 用户加载 AMBER-style mask baseline 配置
- **THEN** 配置 MUST 使用 hard missing mask attention 或等价 baseline
- **AND** 配置 MUST 不启用 reliability bias、prototype alignment 或 full-to-partial KD

#### Scenario: eval patterns 可显式选择
- **WHEN** 用户运行 missing pattern evaluation 并指定 `full missing_gps non_gps_only only_gps random_0.5`
- **THEN** evaluation MUST 分别构造这些 pattern 的 mask
- **AND** report MUST 按 pattern 输出 top-k、loss 和样本数

### Requirement: Numerical and gradient safety checks
本 workflow MUST 提供 focused tests 覆盖 mask dtype、attention zeroing、reliability log safety、KL target normalization、contrastive skip、teacher detach、no-JEPA loss path、all-missing handling、top-k bounds 和 pattern/modalities 一致性。

#### Scenario: focused tests 不读取真实数据
- **WHEN** 运行新增 focused tests
- **THEN** 测试 MUST 使用 synthetic tensor 或 minimal config fixture
- **AND** 测试 MUST 不读取真实 `dataset/`
- **AND** 测试 MUST 不写 tracked checkpoint、cache、logs 或 outputs

#### Scenario: total loss backward 可运行
- **WHEN** 使用 synthetic batch 同时启用 RBMA、prototype alignment 和 online full-to-partial KD
- **THEN** total loss MUST 为有限 tensor
- **AND** 调用 `loss.backward()` MUST 成功
