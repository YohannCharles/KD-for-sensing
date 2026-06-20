## ADDED Requirements

### Requirement: Scene meta-offset whole-model exception
系统 MUST 将 scene-conditioned meta-offset calibration 归类为受控 whole-model exception，同时要求其 scene encoder、support encoder、offset heads、adapters、hypernetwork 和 losses 保持窄模块可测试。该例外 MUST 默认复用 `overlap_k16_s8_stage1` 作为 canonical visual/JEPA 基底，并 MUST 复用共享 batch runtime、`ModelOutput` adaptation、registry 构建和训练策略 metadata，不得复制通用训练/评估循环。

#### Scenario: 注册例外模型
- **WHEN** 实现新增 `scene_conditioned_meta_offset` 或等价 `MODELS` 注册名
- **THEN** OpenSpec、metadata 和 focused tests MUST 声明该模型是 scene-conditioned meta-offset whole-model exception
- **AND** registry build test MUST 覆盖 synthetic config 构建、forward 输出和 `adapt_model_output` 兼容
- **AND** 默认构建 MUST 记录 canonical base variant 为 `overlap_k16_s8_stage1`

#### Scenario: 子组件保持可组合
- **WHEN** 实现新增 ImageOffsetHead、FusionOffsetHead、RadioOffsetHead、SupportSetEncoder 或 HierarchicalHyperNetwork
- **THEN** 这些子组件 MUST 位于窄模块或 registry/factory 可构建路径中
- **AND** 单元测试 MUST 能不构建完整训练 loop 直接验证其输入输出 shape 和 metadata

#### Scenario: 训练策略 metadata
- **WHEN** scene meta-offset 模型完成构建或训练 run 写出 metadata
- **THEN** metadata MUST 记录模型注册名、架构类别、canonical base variant、visual tokenizer type、tokenizer kernel/stride、pooler type、启用模态、scene conditioning 来源、support usage、enabled offset heads、hypernetwork mode、meta method、adapt_modules 和是否消费 reliability/sensitive metadata
- **AND** 缺少这些字段 MUST 被 focused test 或 architecture boundary test 捕获
