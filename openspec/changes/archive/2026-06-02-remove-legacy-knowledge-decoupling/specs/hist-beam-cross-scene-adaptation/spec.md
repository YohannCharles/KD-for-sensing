## MODIFIED Requirements

### Requirement: HiST-Beam 模型变体配置
系统 MUST 提供可通过配置和模型注册表构建的 HiST-Beam fusion 模型能力，用于 DeepSense6G 和 MMW 跨场景快速验证。配置 MUST 能选择 flat source-only、hierarchical source-only、adapter-only、adapter+coarse prototype、adapter+radio-semantic prototype、adapter+path-level physical prototype、target prior/prototype probe、shared physical private residual、history/residual calibration 和 full fine-tuning baseline 变体，并 MUST 默认保持既有 DeepSense6G `image`、`radar`、`gps` 三模态快速验证兼容。系统 MUST NOT 将 `v2_shared_private`、`shared_private`、`v3_decoupled` 或 `decoupled` 作为可构建或默认 HiST-Beam 变体。

#### Scenario: 构建 flat source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v0_flat` 或等价 flat 模式
- **THEN** 系统 MUST 构建普通 64 类 beam classifier
- **AND** 模型输出 MUST 继续兼容现有 beam Top-K 评估流程

#### Scenario: 构建 hierarchical source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v1_hierarchical`
- **THEN** 系统 MUST 构建 coarse head 和 fine head
- **AND** 系统 MUST 不启用旧 shared/private 解耦 loss 或 target adapter

#### Scenario: 拒绝旧 shared-private 解耦变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v2_shared_private`、`shared_private`、`v3_decoupled` 或 `decoupled`
- **THEN** 系统 MUST 拒绝构建该模型或 LOSO run
- **AND** 错误信息 MUST 说明旧简单 shared/private 解耦路线已退役，并指向可用 baseline

#### Scenario: 构建 adapter 和 full fine-tuning 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v4_adapter`、`v5_adapter_proto`、`v6_radio_proto`、`v8_path_proto`、`v8_target_prior_head`、`v9_input_conditioned_target_adaptation` 或 `v6_full_finetune`
- **THEN** 系统 MUST 从非旧解耦 source checkpoint 或显式配置的合法 source checkpoint 初始化 target adaptation run
- **AND** 系统 MUST 按变体选择 adapter 训练、coarse/radio/path prototype adaptation、target prior/prototype probe 或全量 fine-tuning 策略
- **AND** 若工程继续保留 `v6_full_finetune` 配置名，summary MUST 将其标记为 full fine-tuning baseline 或等价 full fine-tuning baseline metadata

#### Scenario: 构建 V6 radio-semantic prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v6_radio_proto`
- **THEN** 系统 MUST 使用 beam_power 派生的 radio-semantic label/prototype 作为 V6 baseline
- **AND** 系统 MUST 不把 V6 radio prototype 静默标记为 V8 path-level physical prototype
- **AND** 系统 MUST NOT 要求旧 `v3_decoupled` source baseline 或旧 shared/private scene loss

#### Scenario: 构建 V8 path-level physical prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v8_path_proto` 或等价 P3-HiST-Beam 模式
- **THEN** 模型 MUST 产生 beam_logits、path_logits、path/prototype 所需表示和可诊断 metadata
- **AND** target adaptation MUST 支持 `proto_type=path`
- **AND** path prototype MUST 作为 semantic anchor 或 condition，而不是直接预测 beam
- **AND** 系统 MUST NOT 要求旧 `v3_decoupled` source baseline 或旧 shared/private scene loss

#### Scenario: 构建 V7 shared physical private residual 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v7_shared_physical_private_residual`
- **THEN** 系统 MUST 构建 shared beam head、physical beamspace head、private adapter、private residual head 和 residual gate
- **AND** 模型 MUST 输出 `logits_shared`、`logits_final`、`delta_logits_private`、`alpha`、`pred_beamspace_power`、shared representation 和 private representation
- **AND** `logits` 与 `beam_logits` MUST 指向 `logits_final` 以保持现有评估入口兼容
- **AND** v7 默认 MUST NOT 启用 history-anchor、读取历史 beam label 或启用旧 scene confusion/private preservation loss

### Requirement: HiST-Beam 训练 loss
系统 MUST 在显式启用 HiST-Beam 时计算当前变体要求的层次化 loss、flat auxiliary loss、radio/path/prototype/residual/target-prior loss 或 full fine-tuning loss。系统 MUST NOT 为旧简单 shared/private 解耦路线计算 orthogonality loss、shared scene confusion loss 或 private scene preservation loss。普通非 HiST 配置 MUST 不受这些 loss 影响。

#### Scenario: 计算 hierarchical loss
- **WHEN** 训练 hierarchical 变体
- **THEN** 系统 MUST 对 coarse logits 计算 coarse CE
- **AND** 系统 MUST 只在真实 coarse group 对应的 fine logits 上计算 fine CE
- **AND** 系统 MUST 按配置权重合成 hierarchical loss

#### Scenario: 计算 flat auxiliary loss
- **WHEN** 配置 `lambda_flat` 大于 0
- **THEN** 系统 MUST 从 beam-level 输出计算 beam class 辅助 loss
- **AND** 该 loss MUST 参与总 loss 以约束最终 beam prediction

#### Scenario: 拒绝旧解耦 loss 权重
- **WHEN** 配置包含旧解耦专属权重 `orthogonality`、`scene_confusion`、`scene_private`、`lambda_orth`、`lambda_scene_c` 或 `lambda_scene_s` 且未处于归档兼容解析场景
- **THEN** 训练配置解析 MUST 拒绝或忽略这些权重并记录清晰迁移信息
- **AND** 总 loss MUST 不包含旧 orthogonality、shared scene confusion 或 private scene preservation 项

#### Scenario: HiST loss 不影响普通模型
- **WHEN** 用户运行非 HiST-Beam 模型或未启用 HiST loss 的配置
- **THEN** 训练流程 MUST 使用既有 beam loss 语义
- **AND** 系统 MUST 不要求模型输出 coarse/fine/shared/private diagnostics

## REMOVED Requirements

### Requirement: Shared-private 表示解耦
**Reason**: 旧路线把知识简单拆为 shared 与 private 分支，并依赖 orthogonality、shared scene confusion 和 private scene preservation 约束；多轮跨场景迁移实验已证明该方法不可行，目标精度长期在约 10% 徘徊。

**Migration**: 使用 `v0_flat`/`v1_hierarchical` 作为 source-only baseline，或使用 image-only legal probe、target prior/prototype、path/radio prototype、V7 physical residual、history/geometry residual calibration 等现行路线。现代路线若仍输出 shared/private 字段，MUST 由各自 residual/prototype/calibration 契约定义语义，不得复用旧简单解耦 loss。

#### Scenario: 旧 shared-private 契约不再要求实现
- **WHEN** 开发者修改 HiST-Beam 模型或 loss
- **THEN** 系统 MUST NOT 要求 coarse head 只读取旧 shared representation、fine head 读取旧 shared/private 组合、private scene classifier 或 shared scene confusion classifier 作为旧简单解耦契约的一部分
