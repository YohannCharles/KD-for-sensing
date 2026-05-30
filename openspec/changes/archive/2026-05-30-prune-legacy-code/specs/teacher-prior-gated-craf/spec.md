## REMOVED Requirements

### Requirement: Teacher reliability registry
**Reason**: teacher reliability registry 只服务于 teacher-prior CRAF/MARF 的 prior gate 和 encoder 初始化，相关架构已退役。
**Migration**: 单模态 teacher checkpoint 继续通过现有 checkpoint/配置路径使用；不再生成 CRAF/MARF teacher prior registry。

#### Scenario: teacher registry 构建入口删除
- **WHEN** 用户查找 teacher-prior CRAF registry 构建流程
- **THEN** 系统 MUST 不再要求提供该入口
- **AND** 文档 MUST 不推荐生成 teacher reliability registry

### Requirement: Prior residual gate
**Reason**: `prior_residual_sigmoid` gate 是 teacher-prior CRAF 专属组件。
**Migration**: 使用当前保留的 fusion 模型和模态选择语义；如需新的 gate，应在新 change 中定义。

#### Scenario: prior residual gate 不再可构建
- **WHEN** 配置请求 `gate_type: prior_residual_sigmoid`
- **THEN** 系统 MUST 拒绝该配置或因 CRAF/MARF 模型不可用而失败
- **AND** 错误信息 MUST 指出该 gate 已不属于支持入口

### Requirement: Teacher encoder initialization
**Reason**: Stage 2 CRAF encoder 初始化随 teacher-prior CRAF 退役。
**Migration**: 普通 checkpoint 加载继续由现有训练/评估流程处理。

#### Scenario: CRAF teacher encoder loader 删除
- **WHEN** 测试或代码尝试调用 CRAF teacher encoder loader
- **THEN** 系统 MUST 不再提供该专属 loader
- **AND** 训练流程 MUST 不读取 teacher-prior registry 来初始化 CRAF encoder

### Requirement: Stage 2 prior-guided fusion training
**Reason**: Stage 2 prior-guided fusion 是 teacher-prior CRAF 主实验流程，已退役。
**Migration**: 使用标准 fusion 或当前保留的模块化 fusion 配置。

#### Scenario: Stage 2 配置不可用
- **WHEN** 用户加载 Stage 2 teacher-init prior CRAF 配置
- **THEN** 配置加载或模型构建 MUST 失败
- **AND** 项目 MUST 不再提供该配置作为推荐入口

### Requirement: Stage 3 selective fine-tuning
**Reason**: Stage 3 selective fine-tuning 只服务于退役的 CRAF 实验阶段。
**Migration**: 标准训练的参数冻结/恢复语义继续由通用训练配置约束。

#### Scenario: Stage 3 配置不可用
- **WHEN** 用户加载 Stage 3 selective fine-tuning CRAF 配置
- **THEN** 系统 MUST 不再构建对应工作流
- **AND** 相关参数组学习率测试 MUST 被删除

### Requirement: Optional reliability KD and counterfactual ablations
**Reason**: reliability KD 与 counterfactual ablation 是 CRAF 研究线内部实验。
**Migration**: KD 使用当前保留的 distillation 类型；反事实 gate 研究需重新提出。

#### Scenario: CRAF ablation 字段不再生效
- **WHEN** 配置包含 CRAF reliability KD 或 counterfactual ablation 字段
- **THEN** 系统 MUST 不把这些字段解释为支持行为
- **AND** 当前保留模型 MUST 不读取这些 CRAF 专属字段

### Requirement: Teacher-prior CRAF diagnostics
**Reason**: gate、prior、residual、teacher load/freeze 状态等 diagnostics 只服务于退役 CRAF/MARF。
**Migration**: 普通训练继续写出通用日志和 metrics。

#### Scenario: teacher-prior diagnostics 不再要求
- **WHEN** 训练完成一个 epoch
- **THEN** 系统 MUST 不要求写出 CRAF prior/residual/gate diagnostics
- **AND** 缺少这些字段 MUST 不影响当前保留工作流验收

### Requirement: Shared teacher prior initialization for MARF
**Reason**: MARF 架构和 teacher prior 接入已退役。
**Migration**: 使用当前保留的 fusion 模型；不再从 teacher registry 初始化 MARF router prior。

#### Scenario: MARF prior 初始化不可用
- **WHEN** 用户配置 `model.student.type: marf_fusion`
- **THEN** 系统 MUST 拒绝该模型类型
- **AND** 不得继续读取 teacher registry 应用 MARF prior

### Requirement: Shared teacher encoder initialization for MARF
**Reason**: MARF encoder loader 随 MARF 架构退役。
**Migration**: 普通 checkpoint 加载不经过 MARF encoder 初始化路径。

#### Scenario: MARF encoder loader 删除
- **WHEN** 代码尝试加载 MARF teacher encoder
- **THEN** 系统 MUST 不再提供该专属路径
- **AND** 相关测试 MUST 被删除

### Requirement: MARF default trainable boundary
**Reason**: MARF routing、anchor fusion、residual adapter 和 subset training 已退役。
**Migration**: 当前保留模型的 trainable boundary 由各自模型和通用 optimizer 构建约束。

#### Scenario: MARF 参数边界不再验证
- **WHEN** 开发者运行项目快速回归
- **THEN** 测试 MUST 不再要求验证 MARF 默认冻结或可训练参数集合
- **AND** `marf_fusion` MUST 不作为可构建模型出现
