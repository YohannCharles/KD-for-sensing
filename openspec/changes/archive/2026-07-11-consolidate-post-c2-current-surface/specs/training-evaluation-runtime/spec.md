## MODIFIED Requirements

### Requirement: 训练配置重构提供 characterization 检查
项目 MUST 为训练编排和配置加载重构提供快速 characterization 检查，覆盖关键输出契约、config load 顺序、十个 retained CLI help 和架构边界。检查 MUST 使用 `kd_mm_beam` 环境，并 MUST 不依赖真实数据、长时间训练或把新 checkpoint 纳入源码。

#### Scenario: 训练短流程 characterization
- **WHEN** 开发者运行训练短流程测试
- **THEN** 测试 MUST 完成 forward、loss、backward、validation、checkpoint 和 artifact 写出
- **AND** 关键输出字段 MUST 保持兼容

#### Scenario: Config load characterization
- **WHEN** 开发者运行 config loading focused tests
- **THEN** 测试 MUST 覆盖实体 YAML、virtual canonical config、migration guard 和命令行覆盖
- **AND** normalization 与 validation MUST 保持兼容

#### Scenario: CLI help characterization
- **WHEN** 开发者运行 CLI help focused tests
- **THEN** 测试 MUST 覆盖 pyproject 声明的十个 retained commands
- **AND** MUST 不要求 project surface doctor、research dashboard 或 research preview

## REMOVED Requirements

### Requirement: apples-to-apples checkpoint 复评
**Reason**: Module-only 复评入口无 src/script/test consumer并重复 evaluate/U-Mask matrix。
**Migration**: 使用 `kd-sensing-evaluate`、U-Mask eval matrix 和显式 checkpoint/config。
#### Scenario: Legacy reevaluation unavailable
- **WHEN** 用户请求 old apples-to-apples module command
- **THEN** module MUST not be current entrypoint

### Requirement: 统一 checkpoint resolver
**Reason**: `utils/checkpoint_resolver.py` 仅专属测试调用，runtime已有 artifact registry/owner-specific resolution。
**Migration**: Current workflow 使用其真实 owner resolver；不新增第四层 facade。
#### Scenario: Orphan resolver removed
- **WHEN** consolidation completes
- **THEN** dedicated resolver module and tests MUST be deleted

### Requirement: eval consistency debug 报告
**Reason**: 对应 one-shot script 已退出 current surface。
**Migration**: 使用 evaluate/U-Mask matrix 和 retained diagnostics。
#### Scenario: Debug script unavailable
- **WHEN** current scripts are enumerated
- **THEN** old eval consistency script MUST not be required
