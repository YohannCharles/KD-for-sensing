## ADDED Requirements

### Requirement: Shared teacher prior initialization for MARF
Teacher registry prior 应用流程 MUST 支持 MARF，同时保持既有 CRAF teacher-prior 行为不变。MARF MUST 能从同一 registry 读取每个启用模态的 prior，并将其作为 router bias 或 diagnostics 使用。

#### Scenario: MARF 从 teacher registry 应用 prior
- **WHEN** 配置设置 `model.student.type: marf_fusion` 且提供 `teacher.registry_path`
- **THEN** 系统 MUST 从 registry 中读取每个启用模态的 prior
- **AND** 系统 MUST 将 prior 写入 MARF router 的 prior buffer 或等价结构
- **AND** 训练 runtime metadata MUST 记录实际应用的 prior

#### Scenario: MARF 缺少 registry 模态时报错
- **WHEN** MARF 启用某个模态但 teacher registry 中缺少该模态
- **THEN** teacher prior 应用流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含缺失模态名称

#### Scenario: CRAF prior 行为不变
- **WHEN** 用户运行既有 `craf_fusion` teacher-prior 配置
- **THEN** 系统 MUST 继续调用 CRAF reliability prior 设置逻辑
- **AND** CRAF 输出的 `gate`、`prior` 和 `residual_logits` diagnostics MUST 保持既有语义

### Requirement: Shared teacher encoder initialization for MARF
Teacher encoder 加载流程 MUST 支持 MARF 的 `encoders` ModuleDict。加载、shape 校验、strict 模式、冻结和日志语义 MUST 与现有 CRAF teacher encoder 初始化一致。

#### Scenario: MARF 加载并冻结 teacher encoder
- **WHEN** MARF 配置设置 `teacher.load_encoders: true` 和 `teacher.freeze_encoders: true`
- **THEN** 系统 MUST 从 teacher registry 加载每个启用模态的 encoder 权重到 MARF
- **AND** 每个成功加载的 encoder 参数 MUST 设置为 `requires_grad=False`
- **AND** router、anchor fusion、residual adapter、feature projection 和 prediction head 参数 MUST 保持 `requires_grad=True`

#### Scenario: MARF teacher key 不匹配可诊断
- **WHEN** teacher checkpoint 中存在不能映射到 MARF encoder 的 key 或 shape 不一致的 tensor
- **THEN** load summary MUST 记录 missing、unexpected 或 shape mismatch
- **AND** strict 模式下系统 MUST 拒绝继续训练

#### Scenario: MARF 只加载启用模态
- **WHEN** MARF 配置只启用部分模态
- **THEN** 系统 MUST 只尝试加载这些启用模态的 teacher encoder
- **AND** 系统 MUST 不要求未启用模态的 teacher registry 项存在

### Requirement: MARF default trainable boundary
MARF teacher-init 主训练 MUST 默认冻结 encoder，并默认只优化 routing、fusion、adapter、projection 和 prediction head 相关参数。

#### Scenario: MARF Stage 2 默认冻结边界
- **WHEN** 用户运行 MARF teacher-init 主训练配置
- **THEN** 成功加载的单模态 encoder MUST 默认冻结
- **AND** MARF router、anchor fusion、residual adapter、feature projection、unimodal head 和 prediction head MUST 可训练
- **AND** 训练日志 MUST 记录每模态 encoder frozen 状态和 trainable parameter count

#### Scenario: MARF 不触发 CRAF Stage 3 强模态默认
- **WHEN** 用户运行 MARF 主训练配置且未显式启用 finetune
- **THEN** 系统 MUST 不默认解冻 GPS 或 mmWave encoder
- **AND** 系统 MUST 不把任何模态硬编码为强 encoder 参数组
