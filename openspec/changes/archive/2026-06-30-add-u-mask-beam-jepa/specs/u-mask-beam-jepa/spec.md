## ADDED Requirements

### Requirement: U-MaskBeamJEPA 模型注册与构建
系统 MUST 提供 `u_mask_beam_jepa` 模型注册名，作为显式 whole-model exception 构建 U-MaskBeamJEPA。该模型 MUST 使用 canonical 模态名称，默认支持 `image`、`radar`、`lidar` 和 `gps`，并 MUST 不注册 `vision` 伪模态或旧式别名。

#### Scenario: registry 构建模型
- **WHEN** 构建流程导入默认组件并请求 `model.primary.type: u_mask_beam_jepa`
- **THEN** `MODELS` registry MUST 返回 U-MaskBeamJEPA 模型实例
- **AND** 模型 metadata MUST 标记 `architecture_category=whole_model_exception`
- **AND** metadata MUST 记录启用模态、是否消费 missing mask、是否消费 reliability metadata 和 ablation flags

#### Scenario: 拒绝未知或重复模态
- **WHEN** 配置为 U-MaskBeamJEPA 提供未知、重复或空模态列表
- **THEN** 系统 MUST 抛出包含非法模态和可用 canonical 模态列表的错误

### Requirement: Forward 输入与输出契约
U-MaskBeamJEPA forward MUST 复用共享 batch/runtime 输入，不要求训练循环新增模型专用分支。forward MUST 接收 batch dict 和可选 `missing_mask`，并 MUST 返回可被 `adapt_model_output` 消费的 dict。

#### Scenario: forward 返回可适配输出
- **WHEN** U-MaskBeamJEPA 使用 synthetic batch 执行 forward
- **THEN** 输出 dict MUST 包含 `logits`
- **AND** `adapt_model_output` MUST 能取得 logits
- **AND** teacher、latent、uncertainty 和 reliability tensor MUST 作为 diagnostics 保留

#### Scenario: 输出字段完整
- **WHEN** forward 成功
- **THEN** 输出 MUST 包含 `teacher_logits`、`u_star`、`mu_B`、`logvar_B`、`modality_reliability` 和 `global_reliability`
- **AND** 输出 MUST 包含 per-modality `modality_mu_B` 和 `modality_logvar_B`
- **AND** `u_star` MUST 从 teacher latent detach 后暴露给损失

### Requirement: 模态 latent 与 mask 语义
U-MaskBeamJEPA MUST 将每个启用模态编码到统一 `d_model` latent，并按固定 canonical 模态顺序堆叠为 `[B, M, d_model]`。missing mask MUST 使用 1/true 表示可用，0/false 表示缺失。

#### Scenario: mask shape 校验
- **WHEN** forward 收到 `missing_mask`
- **THEN** mask shape MUST 为 `[B, M]`
- **AND** `M` MUST 等于启用模态数量
- **AND** shape 不匹配时 MUST 抛出包含实际 shape 和期望 shape 的错误

#### Scenario: 全 0 mask 被拒绝
- **WHEN** 任一样本的 missing mask 显示所有模态缺失
- **THEN** 模型 MUST 抛出清晰错误或使用配置声明的 always-available 模态修正 mask
- **AND** 系统 MUST 不产生 NaN logits 或 NaN loss

### Requirement: Modality reliability head
系统 MUST 为每个启用模态提供 modality reliability head。该 head MUST 从 `z_i: [B, d_model]` 产生 `mu_i_B: [B, d_model]` 和 `logvar_i_B: [B, d_model]` 或 `[B, 1]`，表示该单模态预测的 beam-relevant teacher latent 与不确定性。模态 reliability MUST 以 `sigma_i = mean(softplus(logvar_i_B), dim=-1, keepdim=True)`、`r_i = exp(-sigma_i)` 计算，并以 `[B, M, 1]` 形状暴露。

#### Scenario: per-modality latent prediction 输出
- **WHEN** modality reliability head 收到单模态 latent `z_i`
- **THEN** head MUST 输出 `mu_i_B` 和 `logvar_i_B`
- **AND** 模型 diagnostics MUST 按启用模态顺序暴露 `modality_mu_B` 和 `modality_logvar_B`

#### Scenario: 缺失模态 reliability 置零
- **WHEN** 某个模态在 missing mask 中为缺失
- **THEN** 对应 `modality_reliability` MUST 为 0
- **AND** 缺失模态 MUST 不作为 context encoder 或 fusion attention 的 key/value 参与计算

#### Scenario: reliability 数值有限
- **WHEN** 输入 latent 有限且 mask 至少有一个可用模态
- **THEN** modality reliability MUST 为有限 tensor
- **AND** reliability 值 MUST 位于 `[0, 1]`

### Requirement: Full-modal teacher
系统 MUST 提供 full-modal teacher 分支，消费所有启用模态 latent，加入 learnable modality embedding 和 beam query token，并输出 `u_star: [B, d_model]` 与 `teacher_logits: [B, num_beams]`。

#### Scenario: teacher 使用全模态 latent
- **WHEN** training forward 收到完整 batch
- **THEN** teacher MUST 基于 full-modal latent 计算 `u_star`
- **AND** student missing mask MUST 不移除 teacher 输入中的可用 full-modal supervision

#### Scenario: teacher 可关闭
- **WHEN** ablation 配置 `use_teacher=false`
- **THEN** 模型或损失 MUST 不要求 teacher CE 参与总损失
- **AND** forward MUST 仍能产生 student logits 用于 beam classification

### Requirement: Set context encoder
系统 MUST 提供 permutation-friendly set context encoder。该 encoder MUST 在可用模态 token 上加入 modality embedding 和 reliability embedding，不使用位置编码，并通过 `src_key_padding_mask` 屏蔽缺失模态。

#### Scenario: beam query 输出 context
- **WHEN** context encoder 收到 `[B, M, d_model]` latent、`[B, M]` mask 和 `[B, M, 1]` reliability
- **THEN** encoder MUST 输出 `c_A: [B, d_model]`
- **AND** `c_A` MUST 来自 learnable beam query token 的编码结果

#### Scenario: 缺失 token 不参与 key/value
- **WHEN** 某模态 mask 为缺失
- **THEN** Transformer key padding mask MUST 屏蔽该 token
- **AND** 缺失 token 的数值内容 MUST 不影响其它样本的 attention 结果

### Requirement: Gaussian JEPA predictor
系统 MUST 提供 Gaussian JEPA predictor，从 `c_A: [B, d_model]` 输出 `mu_B: [B, d_model]` 和 `logvar_B: [B, d_model]`。`logvar_B` MUST clamp 到配置的 `[logvar_min, logvar_max]`。

#### Scenario: predictor 输出形状
- **WHEN** predictor 收到 context latent
- **THEN** `mu_B` 和 `logvar_B` shape MUST 均为 `[B, d_model]`
- **AND** `logvar_B` MUST 不小于 `logvar_min` 且不大于 `logvar_max`

#### Scenario: global uncertainty 可关闭
- **WHEN** ablation 配置 `use_global_uncertainty=false`
- **THEN** global reliability MUST 等价于 1
- **AND** loss MAY 忽略 `logvar_B` 的 uncertainty 项但 MUST 保持 backward 可运行

### Requirement: Reliability-gated fusion
系统 MUST 提供 reliability-gated cross-attention fusion。该 fusion MUST 使用 beam query 对可用模态 token 和 `mu_B` token 做 cross-attention，并在 attention logits 上加入 `beta * log(r + eps)` reliability bias。

#### Scenario: global reliability 参与 predicted token
- **WHEN** fusion 构造 `mu_B` token
- **THEN** 该 token reliability MUST 来自 `global_reliability = exp(-mean(softplus(logvar_B)))` 或关闭 global uncertainty 后的 1

#### Scenario: fusion 输出 fused latent
- **WHEN** fusion forward 成功
- **THEN** 输出 fused latent `h` MUST 为 `[B, d_model]`
- **AND** beam prediction head MUST 从 `h` 输出 `logits: [B, num_beams]`

### Requirement: U-MaskBeamJEPA loss
系统 MUST 提供 U-MaskBeamJEPA opt-in loss 扩展，计算 beam classification loss、teacher classification loss、global Gaussian JEPA latent NLL 和 per-modality Gaussian NLL。总损失 MUST 为 `L_beam + lambda_teacher * L_teacher + lambda_jepa_global * L_jepa_global + lambda_modality_nll * L_modality_nll`，并 MUST 支持关闭 teacher 或 JEPA loss。

#### Scenario: loss backward 可运行
- **WHEN** 使用随机 tensor 构造 U-MaskBeamJEPA output 和 beam labels
- **THEN** loss MUST 返回 tensor `loss`
- **AND** 调用 `loss.backward()` MUST 成功
- **AND** scalar diagnostics MUST 包含 `loss_beam`、`loss_teacher`、`loss_jepa_global` 和 `loss_modality_nll` 或对应 disabled 状态

#### Scenario: Gaussian NLL 使用 detached target
- **WHEN** 计算 JEPA latent NLL
- **THEN** target MUST 使用 `u_star.detach()`
- **AND** global JEPA NLL MUST 计算 `0.5 * ((u_star.detach() - mu_B)^2 * exp(-logvar_B) + logvar_B)` 的均值
- **AND** 对每个可用模态 MUST 计算 `0.5 * ((u_star.detach() - mu_i_B)^2 * exp(-logvar_i_B) + logvar_i_B)`
- **AND** loss MUST 只对 `missing_mask_i=1` 的模态求平均，缺失模态不贡献梯度
- **AND** loss 实现 MUST 不更新 full-modal teacher 分支，除非配置显式允许 teacher CE 更新 teacher logits head

### Requirement: Missing mask helper
系统 MUST 提供训练时 missing mask helper，支持随机 mask、pattern mask 和轻量 corruption 占位，并 MUST 不原地修改输入 batch。

#### Scenario: sample_missing_mask 保证至少一个模态可用
- **WHEN** 调用 `sample_missing_mask(batch_size, num_modalities, p_missing, ensure_at_least_one=True)`
- **THEN** 返回 mask MUST 为 `[B, M]`
- **AND** 每个样本 MUST 至少有一个 true/1 可用模态

#### Scenario: always available 模态保持可用
- **WHEN** 调用 `sample_missing_mask` 并传入 `always_available_indices`
- **THEN** 对应模态列 MUST 全部为可用

#### Scenario: corruption 不修改原 batch
- **WHEN** 调用 `apply_modality_corruption(batch, corruption_config)`
- **THEN** 返回的新 batch MAY 包含 vision/image Gaussian noise、zero out、gps noise、lidar/radar placeholder dropout
- **AND** 原 batch tensor MUST 不被原地修改

### Requirement: 训练与评估配置接入
系统 MUST 提供 opt-in U-MaskBeamJEPA 配置，使现有训练入口能完成一个 smoke epoch、保存 checkpoint，并在无 wandb 时通过本地日志记录 metrics。

#### Scenario: smoke 配置使用现有训练入口
- **WHEN** 用户运行 U-MaskBeamJEPA smoke config 的 `kd-sensing-train`
- **THEN** 训练 MUST 通过现有 trainer/runtime 路径执行
- **AND** 系统 MUST 不要求根目录专用训练脚本
- **AND** smoke config MUST 保留 DeepSense6G scene 31 作为快速验证配置

#### Scenario: Scenario 32 正式配置
- **WHEN** 用户加载 `configs/fusion/u_mask_beam_jepa_s32.yaml`
- **THEN** 配置 MUST 使用 DeepSense6G scene 32
- **AND** 配置 MUST 设置 `lambda_teacher=0.5`、`lambda_jepa_global=1.0`、`lambda_modality_nll=0.2`、`logvar_min=-8.0` 和 `logvar_max=4.0`
- **AND** 配置 MUST 设置 missing p_missing 0.5 且保证每个样本至少一个模态可用

#### Scenario: 记录可靠性指标
- **WHEN** U-MaskBeamJEPA 训练 step 完成
- **THEN** metrics MUST 包含 `loss_beam`、`loss_teacher`、`loss_jepa_global`、`loss_modality_nll`、`top1_acc`、`top5_acc`、`mean_modality_reliability` 和 `mean_global_reliability`

#### Scenario: eval 指定 missing pattern
- **WHEN** eval 配置指定 available modalities 或 pattern mask
- **THEN** 模型 forward MUST 使用该 pattern
- **AND** 评估报告 MUST 能区分不同 missing pattern 的结果

### Requirement: Ablation 行为
系统 MUST 为 U-MaskBeamJEPA 提供配置化 ablation flags，并确保关闭分支后模型仍可执行 student beam prediction。

#### Scenario: 关闭 modality uncertainty
- **WHEN** `use_modality_uncertainty=false`
- **THEN** 可用模态 reliability MUST 视为 1
- **AND** 缺失模态 reliability MUST 仍为 0

#### Scenario: fusion type 可切换
- **WHEN** 配置 `fusion_type` 为 `concat_mlp`、`weighted_sum` 或 `reliability_gated_cross_attention`
- **THEN** 模型 MUST 构建对应 fusion 路径
- **AND** 未实现或未知 fusion type MUST 抛出清晰错误

#### Scenario: JEPA loss 可关闭
- **WHEN** `use_jepa_loss=false`
- **THEN** 总损失 MUST 不包含 Gaussian JEPA NLL
- **AND** student logits 的 beam CE MUST 仍可训练模型

### Requirement: Focused tests 与验证命令
本 change MUST 添加 focused tests 覆盖 registry build、synthetic forward、loss backward、mask helper、ablation 开关、metadata 和 architecture boundary。

#### Scenario: focused tests 通过
- **WHEN** 运行 U-MaskBeamJEPA focused tests
- **THEN** 测试 MUST 不读取真实 `dataset/`
- **AND** 测试 MUST 不写 tracked checkpoint、cache、logs 或 outputs

#### Scenario: OpenSpec 校验通过
- **WHEN** 运行 `openspec validate add-u-mask-beam-jepa --strict`
- **THEN** change artifacts MUST 通过严格校验
