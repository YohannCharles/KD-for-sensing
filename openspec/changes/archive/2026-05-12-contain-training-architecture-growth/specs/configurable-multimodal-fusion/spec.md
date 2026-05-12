## ADDED Requirements

### Requirement: 高级 fusion 方法配置 overlay
CRAF、MARF、G2D 和后续高级 fusion 方法配置 MUST 支持通过 base 配置、method overlay 和 ablation overlay 组合生成或解析。推荐配置路径 MUST 避免为每个方法、场景和 ablation 复制完整 data/model/training/output 配置；实体 YAML MAY 继续存在，但新增推荐路径 MUST 优先复用共享配置语义。

#### Scenario: 生成 G2D 方法配置
- **WHEN** 开发者加载推荐的 G2D fusion 配置路径或 recipe
- **THEN** 系统 MUST 从共享 fusion base 和 G2D method overlay 解析出完整配置
- **AND** `final_config.yaml` MUST 写出完整解析后的 data、model、loss、distillation、training 和 output 字段

#### Scenario: 生成 CRAF 或 MARF ablation 配置
- **WHEN** 开发者加载 CRAF 或 MARF 的 ablation 配置路径或 recipe
- **THEN** 系统 MUST 复用共享 method overlay
- **AND** ablation overlay MUST 只表达与 baseline 方法不同的字段，例如 prior、residual、subset training、counterfactual 或 gate 设置

#### Scenario: 场景选择不复制方法配置
- **WHEN** 用户通过命令行或配置字段切换 DeepSense6G scene
- **THEN** 系统 MUST 保持方法 overlay 不变
- **AND** scene 信息 MUST 只通过 dataset scene、输出 scene 目录或运行 metadata 表达

### Requirement: 高级 fusion 实体 YAML 兼容
现有 `configs/fusion/*.yaml` 高级方法实体配置 MUST 继续可加载，并 MUST 与 overlay 解析语义兼容。若同一路径同时存在实体 YAML 和虚拟 overlay 规则，实体 YAML MUST 优先；训练产物仍 MUST 保存完整 `final_config.yaml` 以保证复现。

#### Scenario: 现有实体 YAML 优先
- **WHEN** 用户加载一个已经存在的 `configs/fusion/*.yaml` 文件
- **THEN** 系统 MUST 使用该实体 YAML 的内容
- **AND** 不得用虚拟 overlay 规则覆盖实体 YAML 中显式配置的字段

#### Scenario: overlay 与实体配置语义一致
- **WHEN** 一个高级 fusion 方法同时有实体 YAML 和等价 overlay 入口
- **THEN** 两种入口解析后的关键语义 MUST 一致，包括 task、modalities、model type、loss/distillation type、training schedule 和 run_name
- **AND** 差异字段 MUST 是显式记录的兼容或实验差异

#### Scenario: 配置矩阵测试覆盖 overlay
- **WHEN** 开发者运行 fusion 配置矩阵测试
- **THEN** 测试 MUST 覆盖高级方法 overlay 入口的可加载性和关键字段
- **AND** 测试 MUST 验证现有实体 YAML 仍按兼容语义加载
