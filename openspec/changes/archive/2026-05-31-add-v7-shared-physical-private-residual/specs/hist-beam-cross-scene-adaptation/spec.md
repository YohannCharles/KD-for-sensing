## MODIFIED Requirements

### Requirement: HiST-Beam 模型变体配置
系统 MUST 提供可通过配置和模型注册表构建的 HiST-Beam fusion 模型能力，用于 DeepSense6G 和 MMW 跨场景快速验证。配置 MUST 能选择 flat source-only、hierarchical source-only、shared-private、decoupled shared-private、adapter-only、adapter+coarse prototype、adapter+radio-semantic prototype、adapter+path-level physical prototype、shared physical private residual 和 full fine-tuning baseline 变体，并 MUST 默认保持既有 DeepSense6G `image`、`radar`、`gps` 三模态快速验证兼容。

#### Scenario: 构建 flat source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v0_flat` 或等价 flat 模式
- **THEN** 系统 MUST 构建普通 64 类 beam classifier
- **AND** 模型输出 MUST 继续兼容现有 beam Top-K 评估流程

#### Scenario: 构建 hierarchical source-only 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v1_hierarchical`
- **THEN** 系统 MUST 构建 coarse head 和 fine head
- **AND** 系统 MUST 不启用 shared/private 解耦 loss 或 target adapter

#### Scenario: 构建 shared-private 解耦变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v3_decoupled`
- **THEN** 模型 MUST 产生 shared representation、private representation、coarse logits、fine logits 和 beam-level prediction
- **AND** 训练 MUST 能启用 orthogonality、shared scene confusion 和 private scene preservation loss

#### Scenario: 构建 adapter 和 full fine-tuning 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v4_adapter`、`v5_adapter_proto`、`v6_radio_proto`、`v8_path_proto` 或 `v6_full_finetune`
- **THEN** 系统 MUST 从 source checkpoint 初始化 target adaptation run
- **AND** 系统 MUST 按变体选择 adapter 训练、coarse/radio/path prototype adaptation 或全量 fine-tuning 策略
- **AND** 若工程继续保留 `v6_full_finetune` 配置名，summary MUST 将其标记为 V7 full fine-tuning baseline 或等价 full fine-tuning baseline metadata

#### Scenario: 构建 V6 radio-semantic prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v6_radio_proto`
- **THEN** 系统 MUST 使用 beam_power 派生的 radio-semantic label/prototype 作为 V6 baseline
- **AND** 系统 MUST 不把 V6 radio prototype 静默标记为 V8 path-level physical prototype

#### Scenario: 构建 V8 path-level physical prototype 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v8_path_proto` 或等价 P3-HiST-Beam 模式
- **THEN** 模型 MUST 产生 beam_logits、path_logits、shared representation `c` 和 private representation `s`
- **AND** target adaptation MUST 支持 `proto_type=path`
- **AND** path prototype MUST 作为 semantic anchor 或 condition，而不是直接预测 beam

#### Scenario: 构建 V7 shared physical private residual 变体
- **WHEN** 用户配置 HiST-Beam 变体为 `v7_shared_physical_private_residual`
- **THEN** 系统 MUST 构建 shared beam head、physical beamspace head、private adapter、private residual head 和 residual gate
- **AND** 模型 MUST 输出 `logits_shared`、`logits_final`、`delta_logits_private`、`alpha`、`pred_beamspace_power`、shared representation 和 private representation
- **AND** `logits` 与 `beam_logits` MUST 指向 `logits_final` 以保持现有评估入口兼容
- **AND** v7 默认 MUST NOT 启用 history-anchor 或读取历史 beam label

## ADDED Requirements

### Requirement: V7 shared physical private residual forward contract
V7 模型 MUST 让 shared 分支独立预测 beam，并让 private 分支只产生 gated residual correction。private residual MUST NOT 作为完整 beam prediction 单独训练或评估为主输出。

#### Scenario: final logits 由 shared 加 residual 得到
- **WHEN** V7 forward 接收有效 multimodal batch
- **THEN** `logits_final` MUST 等于 `logits_shared + alpha * delta_logits_private`
- **AND** `alpha` 第一版 MUST 支持 shape `[B, H, 1]` 或可 broadcast 到 beam class 维

#### Scenario: shared 分支独立可评估
- **WHEN** evaluation 读取 V7 输出
- **THEN** 系统 MUST 能仅使用 `logits_shared` 计算 Top-K、beam power loss 和 NRP
- **AND** shared-only 指标 MUST 与 final 指标分开记录

#### Scenario: private residual 不作为完整预测
- **WHEN** V7 训练或评估运行
- **THEN** 系统 MUST NOT 把 `delta_logits_private` 直接作为 beam classifier 主 logits
- **AND** 训练 MUST 对 residual magnitude 或 gate 增加约束，防止 private 分支偷走完整预测任务

### Requirement: V7 source training losses
系统 MUST 在 V7 source training 中计算 shared hard CE、final hard CE、beamspace soft KL、physical head KL、residual L2、gate L1 和 shared/private difference loss，并按配置权重合成 total loss。

#### Scenario: 使用 beamspace_power_label 计算 shared physical loss
- **WHEN** V7 source batch 包含有效 `beamspace_power_label`
- **THEN** 系统 MUST 使用 `log_softmax(logits_shared / T)` 与 BSP target 计算 KL loss
- **AND** 系统 MUST 使用 `pred_beamspace_power` 与 BSP target 计算 physical head KL loss

#### Scenario: warmup 阶段禁用 private residual
- **WHEN** 当前 epoch 小于 `training.shared_warmup_epochs`
- **THEN** V7 training MUST 令 final prediction 等价于 shared prediction
- **AND** total loss MUST 不包含 final residual、residual L2、gate L1 或 private residual 相关项

#### Scenario: BSP 缺失时不静默训练物理 loss
- **WHEN** V7 source batch 缺少有效 `beamspace_power_label`
- **THEN** 系统 MUST 按配置拒绝训练或将 physical loss 标记为 unavailable
- **AND** diagnostics MUST 记录不可用原因

### Requirement: V7 target private residual adaptation
系统 MUST 支持从 V7 source checkpoint 启动 target adaptation，并在默认策略中冻结 shared backbone、shared heads 和 physical head，只训练 target private adapter、private residual head、residual gate 和配置允许的 norm affine 参数。

#### Scenario: V7 adaptation 冻结 shared 参数
- **WHEN** 用户应用 adaptation strategy `v7_private_residual`
- **THEN** modality encoders、fusion transformer、shared branch、shared beam head 和 physical head 参数 MUST `requires_grad=false`
- **AND** trainable parameter summary MUST 反映实际白名单参数比例

#### Scenario: V7 adaptation loss 不使用 target physical oracle
- **WHEN** target labeled adaptation batch 包含 hard beam label 和 target-side `beamspace_power_label`
- **THEN** 默认 adaptation loss MUST 只使用 final hard CE、residual L2 和 gate L1
- **AND** 系统 MUST NOT 使用 target-side BSP 对 shared 分支进行训练反传

#### Scenario: V7 不使用历史 label
- **WHEN** V7 source training 或 target adaptation 运行
- **THEN** 模型输入 kwargs MUST NOT 要求 `input_beam_batch` 或 `last_beam_batch`
- **AND** leakage diagnostics MUST 标记 `uses_input_beam_as_model_input=false`

### Requirement: V7 evaluation metrics and artifacts
系统 MUST 在 V7 evaluation、adapted target_test evaluation 和 LOSO summary 中输出 shared-only 与 final prediction 的对比指标，以及 gate、residual 和 physical alignment 诊断。

#### Scenario: metrics 包含 shared 和 final 指标
- **WHEN** V7 evaluation 完成
- **THEN** metrics MUST 包含 `shared_top1`、`shared_top3`、`final_top1`、`final_top3`
- **AND** 若 beam power vector 可用，metrics MUST 包含 `shared_beam_loss_db`、`final_beam_loss_db`、`shared_nrp` 和 `final_nrp`

#### Scenario: metrics 包含 residual 诊断
- **WHEN** V7 evaluation 完成
- **THEN** metrics MUST 包含 `alpha_mean`、`alpha_std` 和 `delta_norm`
- **AND** 若 BSP target 可用，metrics MUST 包含 `phys_kl`

#### Scenario: predictions 标明 final 和 shared 输出
- **WHEN** 系统写出 V7 target_test predictions
- **THEN** predictions MUST 至少包含 sample id、scene、true beam、final predicted beam、shared predicted beam、final top-k、shared top-k 和 variant metadata
- **AND** predictions MUST 标明样本来自 `target_test`

#### Scenario: LOSO summary 汇总 V7 字段
- **WHEN** V7 source-only 或 adapted run 写入 LOSO summary
- **THEN** summary MUST 包含 variant、target_scene、budget、seed、shared/final accuracy、alpha/residual 诊断和 physical KL
- **AND** summary MUST 能与 v3/v4/v6/v8/full-finetune baseline 横向比较
