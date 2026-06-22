## ADDED Requirements

### Requirement: YAML 解析使用项目依赖
配置加载 MUST 使用项目声明的 YAML 依赖解析 YAML。系统 MUST 不维护手写 YAML 子集 parser 或 optional-yaml fallback 作为当前配置语义的一部分。

#### Scenario: 加载 YAML 配置
- **WHEN** 用户加载实体 YAML、virtual config base 或 manifest YAML
- **THEN** 系统 MUST 使用 `pyyaml` 安全解析配置
- **AND** 解析结果 MUST 继续支持当前实体 YAML、`_base_`、命令行 override 和 config normalization 流程

#### Scenario: 删除手写 YAML fallback
- **WHEN** 项目环境缺失 `pyyaml`
- **THEN** 配置加载 MUST 失败并暴露依赖缺失
- **AND** 系统 MUST 不回退到手写 YAML 子集 parser

### Requirement: Canonical recipe 小层可合并
只包装少量常量表、没有独立 public API、没有多个真实实现且只被 `canonical.py` 消费的 recipe/dataclass 文件 MAY 合并到 `canonical.py` 或单一 owner。合并 MUST 保持 virtual config 关键语义、实体 YAML 优先和命令行覆盖顺序。

#### Scenario: 合并 fusion training recipe
- **WHEN** canonical fusion training defaults 从独立 dataclass 文件迁入 owner
- **THEN** `training_overrides()` 或等价逻辑 MUST 对 strong/lightweight、image-radar 和一般 fusion 生成与变更前一致的关键字段
- **AND** `tests/test_config_load_characterization.py` 或等价 focused test MUST 覆盖该行为

#### Scenario: 合并 objective overlay recipe
- **WHEN** objective overlay recipe 常量迁入 owner
- **THEN** occlusion、position 和 multitask virtual config MUST 保持 objective、dataset target、loss、training metric 和 output run name 语义
- **AND** 未知 overlay MUST 继续给出清晰错误

### Requirement: 配置瘦身不重新实体化
删除手写 parser 或合并 recipe 小层时，项目 MUST 不通过新增实体 YAML 或复制默认表来恢复同一复杂度。

#### Scenario: 不新增重复实体配置
- **WHEN** recipe 小层被合并
- **THEN** 实现 MUST 不新增与 recipe 等价的实体 YAML 来弥补删除
- **AND** final/resolved config MUST 继续保存完整解析结果
