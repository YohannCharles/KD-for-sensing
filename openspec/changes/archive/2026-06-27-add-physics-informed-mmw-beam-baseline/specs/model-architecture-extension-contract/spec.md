## ADDED Requirements

### Requirement: Physics-informed MMW whole-model exception
系统 MUST 将 `pinn_multimodal_beam` 登记为显式 whole-model exception。该模型 MUST 说明不能仅通过 `modular_sequence` encoder/core/head 表达的原因，MUST 复用共享 batch/runtime 和 `ModelOutput` 适配路径，MUST 提供训练策略 metadata、registry build、synthetic forward、loss/backward 和架构摘要 focused tests。

#### Scenario: PINN 模型例外可构建
- **WHEN** 构建流程导入默认组件并解析 `model.primary.type: pinn_multimodal_beam`
- **THEN** `MODELS` registry MUST 返回对应模型实例
- **AND** 该模型 forward 输出 MUST 能被 `adapt_model_output` 消费
- **AND** 训练循环 MUST 不需要新增模型专用 forward 分支

#### Scenario: PINN 模型 metadata 最小字段
- **WHEN** `pinn_multimodal_beam` 被构建或训练
- **THEN** `training_strategy_metadata()` 或等价 metadata MUST 记录模型注册名、architecture category、enabled modalities、physics branch、array type、codebook source、loss weights 和 sensitive physical supervision usage
- **AND** metadata MUST 记录该模型是否消费 CSI、path label、beam power 或 reliability metadata

#### Scenario: 架构摘要覆盖 PINN 模型
- **WHEN** 模型架构摘要检查 `pinn_multimodal_beam`
- **THEN** summary MUST 记录 total/trainable params、注册名、whole-model exception 类别、启用模态和 physics branch 配置
- **AND** 如果内部 path head 或 channel synthesizer 无法自动分组，summary MUST 至少保留 unknown component role 和参数统计
