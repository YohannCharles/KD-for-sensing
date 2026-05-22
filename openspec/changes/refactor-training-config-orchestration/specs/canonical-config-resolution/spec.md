## MODIFIED Requirements

### Requirement: config/io 职责收敛
配置加载流程 MUST 将 config source 解析、命令行 overlay、normalization pipeline、migration guard、dataset-specific rules 和 schema validation 拆到明确 helper。`config/io.py` MAY 作为入口协调这些 helper，但 MUST 不手写 canonical overlay、dataset 专属规则、removed feature guard 或 objective 专属默认表的主要实现。

#### Scenario: 加载实体配置
- **WHEN** 用户加载磁盘上存在的 YAML 配置
- **THEN** `config/io.py` MUST 读取实体文件并应用覆盖、默认补全和校验
- **AND** MUST 不进入 canonical recipe 生成路径
- **AND** 后续 normalization 和 validation MUST 通过明确 helper 执行

#### Scenario: 加载 virtual 配置
- **WHEN** 用户加载缺失但合法的 canonical virtual 配置路径
- **THEN** `config/io.py` MUST 调用 canonical recipe 入口获得基础配置
- **AND** 后续覆盖、默认补全和校验流程 MUST 与实体配置一致

#### Scenario: dataset 专属规则位于 dataset rule helper
- **WHEN** 用户加载 Raymobtime s008、DeepSense6G snapshot 或其它 dataset 专属配置
- **THEN** dataset 专属配置约束 MUST 由 dataset rule helper 或等价 validation helper 执行
- **AND** `config/io.py` MUST 不直接维护该 dataset 的完整业务规则

#### Scenario: migration guard 独立于 io 入口
- **WHEN** 用户配置已删除的 image motion profile、image cache、legacy encoder 或其它迁移拒绝项
- **THEN** 系统 MUST 由 migration guard 或等价 helper 抛出清晰错误
- **AND** `config/io.py` MUST 不直接维护所有已删除选项的完整拒绝逻辑

#### Scenario: normalization 顺序可测试
- **WHEN** 开发者对实体配置、virtual canonical 配置、snapshot 配置或 Raymobtime 配置运行 config load 等价测试
- **THEN** 测试 MUST 能验证 source、overlay、normalization 和 validation 的执行顺序
- **AND** 命令行覆盖 MUST 继续在配置生成后生效，并在必要的 runtime requirement 校验前被考虑
