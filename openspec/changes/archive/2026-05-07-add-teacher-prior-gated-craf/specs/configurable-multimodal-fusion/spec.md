## ADDED Requirements

### Requirement: Teacher-prior CRAF 配置入口
Fusion 配置 MUST 支持 teacher-prior CRAF Stage 2、Stage 3 和消融实验入口。新增配置 MUST 使用现有 fusion 数据字段、固定模态顺序和 CRAF 显式模型类型，不得改变 legacy fusion 默认行为。

#### Scenario: Stage 2 主实验配置可加载
- **WHEN** 用户加载 Stage 2 teacher-init prior CRAF 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** 配置 MUST 使用 `model.student.type: craf_fusion`
- **AND** 配置 MUST 启用 `["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 提供 teacher registry 路径、`gate_type: prior_residual_sigmoid`、prior regularization 权重和 frozen encoder 策略

#### Scenario: Stage 3 主实验配置可加载
- **WHEN** 用户加载 Stage 3 selective fine-tuning CRAF 配置
- **THEN** 配置 MUST 设置 Stage 2 checkpoint 加载路径
- **AND** 配置 MUST 提供 `finetune.unfreeze_modalities` 和 `finetune.freeze_modalities`
- **AND** 配置 MUST 提供 Stage 3 参数组学习率

#### Scenario: Teacher-prior 消融配置可加载
- **WHEN** 用户加载 teacher-init no-prior、prior random-encoder、teacher-init fixed-prior 或 teacher-init prior-residual 消融配置
- **THEN** 每个配置 MUST 明确 `teacher.load_encoders` 和 gate 类型
- **AND** 每个配置 MUST 使用同一场景、同一 split、同一模态集合和同一基础训练超参数，除消融目标字段外不得隐式改变实验条件

### Requirement: CRAF gate 类型配置
`craf_fusion` 配置 MUST 支持 `none`、`fixed_prior`、`prior_residual_sigmoid` 和旧 gate 类型。每种 gate 类型 MUST 有清晰的 mask 语义和 diagnostics 行为。

#### Scenario: gate none
- **WHEN** 配置设置 `gate_type: none`
- **THEN** CRAF MUST 对所有可用模态使用 gate 1
- **AND** 不可用模态 MUST 仍被 mask 排除

#### Scenario: gate fixed prior
- **WHEN** 配置设置 `gate_type: fixed_prior`
- **THEN** CRAF MUST 使用配置或 teacher registry 中的 prior 作为 gate
- **AND** gate MUST 不引入可学习 residual

#### Scenario: gate prior residual sigmoid
- **WHEN** 配置设置 `gate_type: prior_residual_sigmoid`
- **THEN** CRAF MUST 构建 PriorResidualGate
- **AND** 配置 MUST 能控制 `min_gate`、hidden dim、confidence feature 和 residual 零初始化

### Requirement: Legacy fusion 配置不变
新增 teacher-prior CRAF 配置 MUST 不改变既有 fusion、token transformer、CRAF stabilized 和 fixed prior sanity 配置的解析结果。

#### Scenario: legacy fusion 仍按 image+radar 运行
- **WHEN** 用户加载既有 `configs/fusion/no_kd.yaml` 或 canonical image+radar fusion 配置
- **THEN** 配置 MUST 继续构建 legacy fusion teacher/student
- **AND** 配置 MUST 不自动启用 teacher registry、prior residual gate 或 selective finetune

#### Scenario: 已有 fixed prior sanity 保持语义
- **WHEN** 用户加载 `craf_all_modalities_fixed_prior_sanity` 配置
- **THEN** 配置 MUST 继续使用 fixed prior gate
- **AND** 配置 MUST 不自动加载 teacher encoder
