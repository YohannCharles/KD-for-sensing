## ADDED Requirements

### Requirement: Internal import paths are not compatibility commitments
未被 README/current docs、`pyproject.toml`、current specs、inventory public surface 或 focused tests 明确保护的 import path MUST 视为 internal。Internal path MAY 删除、移动或合并，项目 MUST 不新增 compatibility wrapper、lazy re-export 或 package barrel 来维持旧路径。

#### Scenario: Internal wrapper removal
- **WHEN** 一个模块只 re-export owner symbols、只服务历史 import 或只有单调用点
- **THEN** 本 change MAY 删除该模块或合并回 owner
- **AND** 内部调用方 MUST 直接导入真实 owner module

### Requirement: Retired-route tests use one guard surface
退役路线防回流测试 MUST 优先使用集中 retired-route 清单和参数化测试，而不是为每条退役路线保留独立测试文件和专用 fixture。

#### Scenario: Dedicated tombstone test consolidation
- **WHEN** 一个测试文件只验证旧 CLI、旧 config、旧 module path 或旧 registry name 不存在
- **THEN** 该测试 MUST 合并到集中 retired-route guard
- **AND** 合并后 MUST 继续覆盖旧名称不会出现在 current registry、config virtual path 或 pyproject scripts 中
