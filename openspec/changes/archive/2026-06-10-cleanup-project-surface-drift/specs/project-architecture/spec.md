## ADDED Requirements

### Requirement: OpenSpec 当前规范不得保留脚手架占位
当前 `openspec/specs/` 中的 spec MUST 具备真实 Purpose 和可理解的需求文本。归档 change 产生的 `TBD`、空泛占位或未替换模板文本 MUST 在进入当前规范后被修复，架构边界测试 MUST 能发现这类漂移。

#### Scenario: 当前 spec purpose 可读
- **WHEN** 开发者运行架构边界检查或 OpenSpec hygiene 检查
- **THEN** 当前 specs 的 Purpose MUST 是描述 capability 边界的真实文本
- **AND** Purpose MUST 不包含 `TBD`、未替换模板提示或归档脚手架说明

#### Scenario: 新归档规范进入当前面
- **WHEN** 一个 change 被归档并生成或修改 `openspec/specs/` 下的当前 spec
- **THEN** 归档后的 spec MUST 通过 OpenSpec 校验和项目架构 hygiene 检查
- **AND** 若归档工具留下占位 Purpose，开发者 MUST 在同一清理批次修复

### Requirement: 架构 guardrail 必须匹配真实支持面
架构边界测试、inventory 文档和当前支持入口 MUST 使用同一套项目表面定义。新增、迁移或删除配置、脚本和公开入口时，项目 MUST 同步更新 guardrail、inventory 和引用文档，不得通过过宽阈值掩盖真实漂移。

#### Scenario: 配置数量 guardrail 更新
- **WHEN** `configs/fusion/` 的当前支持 YAML 集合发生变化
- **THEN** 架构边界测试中的数量阈值或 allowlist MUST 与 inventory 中的分类一致
- **AND** 测试 MUST 继续限制根目录无限增长

#### Scenario: 脚本 allowlist 更新
- **WHEN** shell orchestration、thin CLI alias 或 research diagnostic 脚本引用的配置路径变化
- **THEN** 脚本、inventory 和测试 allowlist MUST 同步更新
- **AND** 当前脚本 MUST 不引用不存在的配置文件作为默认入口

### Requirement: 大规模表面清理必须有快速验收
项目 MUST 为大规模表面清理提供快速验收命令，覆盖 OpenSpec 校验、架构边界、CLI help 和被修改入口的引用一致性。所有项目相关 Python 验收 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 清理实现后的快速验收
- **WHEN** 支持面清理实现完成
- **THEN** 开发者 MUST 运行 `openspec validate cleanup-project-surface-drift --strict`
- **AND** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 修改 CLI 或脚本入口后验收
- **WHEN** 清理实现修改 console script、shell orchestration 或可视化入口
- **THEN** 开发者 MUST 运行对应 `--help` 或无副作用 smoke 检查
- **AND** 检查 MUST 不读取真实 dataset、不启动训练、不写入新的源码内产物
