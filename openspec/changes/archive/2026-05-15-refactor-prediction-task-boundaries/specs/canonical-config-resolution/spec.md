## ADDED Requirements

### Requirement: canonical overlay recipe 化
canonical fusion 配置和 advanced overlay 生成 MUST 由可审查的 recipe/table 驱动。objective、G2D、CRAF、MARF 和通用 fusion overlay MUST 按职责拆分定义，`build_virtual_config()` 入口 MUST 只负责路径识别、recipe 查找和应用。

#### Scenario: 既有 canonical 路径生成语义不变
- **WHEN** 用户加载既有 virtual canonical fusion 路径
- **THEN** 系统 MUST 通过 recipe 生成与变更前等价的关键配置语义
- **AND** experiment name、task、modalities、student/teacher model、distillation、loss、training 和 output run name MUST 保持兼容

#### Scenario: objective overlay recipe
- **WHEN** 用户加载 `configs/fusion/gps_mmwave_occlusion_no_kd.yaml`
- **THEN** 系统 MUST 通过 objective recipe 生成 `experiment.objective: occlusion`
- **AND** recipe MUST 启用 occlusion target、occlusion head、objective loss 和对应 early stopping 默认值

#### Scenario: advanced overlay recipe 错误可诊断
- **WHEN** 用户加载未知 advanced overlay 路径
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 列出可用 overlay recipe 名称

### Requirement: config/io 职责收敛
配置加载流程 MUST 将 virtual config 生成、命令行 overlay、迁移拒绝、objective 默认补全和 schema 校验拆到明确 helper。`config/io.py` MAY 作为入口协调这些 helper，但 MUST 不手写 canonical overlay 细节。

#### Scenario: 加载实体配置
- **WHEN** 用户加载磁盘上存在的 YAML 配置
- **THEN** `config/io.py` MUST 读取实体文件并应用覆盖、默认补全和校验
- **AND** MUST 不进入 canonical recipe 生成路径

#### Scenario: 加载 virtual 配置
- **WHEN** 用户加载缺失但合法的 canonical virtual 配置路径
- **THEN** `config/io.py` MUST 调用 canonical recipe 入口获得基础配置
- **AND** 后续覆盖、默认补全和校验流程 MUST 与实体配置一致
