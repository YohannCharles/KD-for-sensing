## ADDED Requirements

### Requirement: 维护上下文索引测试 helper 私有化
项目健康护栏 SHALL 使用测试私有 helper 读取和验证维护上下文索引。架构边界测试 MUST 不长期内联大段 YAML schema validation 和 projection logic；helper MUST 不成为 runtime API。

#### Scenario: 架构测试通过 helper 读取索引
- **WHEN** `tests/test_architecture_boundaries.py` 需要 entrypoint allowlist、hotspot budget 或 retired route token
- **THEN** 测试 MUST 通过测试私有 helper 获取这些数据
- **AND** 测试文件 MUST 不重新维护与 helper 重复的大段 schema validation 逻辑

#### Scenario: runtime 不导入测试 helper
- **WHEN** 开发者导入 `kd_sensing` 或运行训练/评估 CLI
- **THEN** runtime MUST 不导入 `tests.helpers.maintainer_context` 或等价测试 helper
- **AND** helper MUST 不出现在 README 推荐 runtime 入口中

### Requirement: pyproject 和 maintainer index 双向一致
项目健康护栏 SHALL 验证 `pyproject.toml` 的 `[project.scripts]` 与维护上下文索引中的 `governance.entrypoints.package_cli` 双向一致。新增、删除或重命名 console script MUST 同步更新索引。

#### Scenario: pyproject 新增脚本但索引缺失
- **WHEN** `[project.scripts]` 出现新的 `kd-sensing-*` console script
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求在 `docs/maintainer_context_index.yaml` 中登记名称、target 和 lifecycle

#### Scenario: 索引登记脚本但 pyproject 缺失
- **WHEN** 索引 `package_cli` 登记的 console script 不存在于 `pyproject.toml`
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求恢复 pyproject 声明或删除索引登记
