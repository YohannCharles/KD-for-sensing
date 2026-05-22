## MODIFIED Requirements

### Requirement: 高级 fusion 实体 YAML 兼容
高级 fusion 配置 MUST 支持实体 YAML 与 virtual overlay 两种入口，但实体 YAML 不再作为长期保留要求。仍保留的 `configs/fusion/*.yaml` 高级方法实体配置 MUST 继续可加载，并 MUST 与 overlay 解析语义兼容。若同一路径同时存在实体 YAML 和 virtual overlay 规则，实体 YAML MUST 优先；若实体 YAML 已被删除，等价 virtual overlay MUST 生成完整配置，训练产物仍 MUST 保存完整 `final_config.yaml` 以保证复现。

#### Scenario: 保留实体 YAML 优先
- **WHEN** 用户加载一个仍存在的 `configs/fusion/*.yaml` 文件
- **THEN** 系统 MUST 使用该实体 YAML 的内容
- **AND** 不得用 virtual overlay 规则覆盖实体 YAML 中显式配置的字段

#### Scenario: overlay 与实体配置语义一致
- **WHEN** 一个高级 fusion 方法同时有实体 YAML 和等价 overlay 入口
- **THEN** 两种入口解析后的关键语义 MUST 一致，包括 task、modalities、model type、loss/distillation type、training schedule 和 run_name
- **AND** 差异字段 MUST 是显式记录的兼容或实验差异

#### Scenario: 删除冗余实体 YAML 后 overlay 接管
- **WHEN** 一个高级 fusion 实体 YAML 已确认可由 overlay 无损生成并从源码删除
- **THEN** 用户加载对应声明支持的配置路径时 MUST 仍得到完整配置
- **AND** 运行 artifact MUST 不依赖被删除 YAML 文件

#### Scenario: 配置矩阵测试覆盖 overlay
- **WHEN** 开发者运行 fusion 配置矩阵测试
- **THEN** 测试 MUST 覆盖高级方法 overlay 入口的可加载性和关键字段
- **AND** 测试 MUST 验证仍保留的实体 YAML 按兼容语义加载
