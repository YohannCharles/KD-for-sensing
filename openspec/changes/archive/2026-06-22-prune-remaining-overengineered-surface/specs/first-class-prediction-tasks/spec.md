## ADDED Requirements

### Requirement: Objective metadata 可合并但行为不可变
预测目标的默认 metric、metric mode、available metrics、alias、history fields、TensorBoard scalars 和 runtime metadata MAY 合并到单一 owner 表或 helper。合并 MUST 保持每个 objective 的公开字段名和校验行为不变。

#### Scenario: objective metric alias 保持兼容
- **WHEN** 用户配置 beam、occlusion、position、multitask、current beam、LOS、link、selection multitask 或 JEPA objective 的 early stopping alias
- **THEN** alias MUST 解析为与变更前相同的 canonical metric
- **AND** metric mode MUST 保持一致

#### Scenario: TensorBoard scalar 保持隔离
- **WHEN** 当前 objective 为 beam、current beam selection、LOS、link、selection multitask 或 JEPA
- **THEN** objective metadata MUST 输出该 objective 对应的 TensorBoard scalar 集合
- **AND** 不属于该 objective 的 scalar MUST 不被错误写入

### Requirement: Objective 小表不拆成伪 registry
Objective metadata 合并 MUST 不新增 registry、factory、adapter 或多文件常量拆分来替代现有小表。新增 objective 才需要通过新的 OpenSpec change 扩展 metadata 表。

#### Scenario: 合并 history 和 registry 常量
- **WHEN** `_DEFAULT_METRICS`、metric aliases、history fields 或 TensorBoard scalar 表被合并
- **THEN** 调用方 MUST 继续通过 objective metadata owner 查询
- **AND** trainer、validator 和 tensorboard logging MUST 不维护独立重复表
