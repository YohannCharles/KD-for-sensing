## ADDED Requirements

### Requirement: Undocumented internal import paths may be removed
项目 MUST 将 public surface 限定为 README/current docs 推荐入口、pyproject console scripts、current OpenSpec 明确 public owner、inventory 登记的 public facade、registry/config 构建入口和 focused tests 明确保护的路径。未登记为 public surface 的内部 import path MAY 被删除、移动或合并。

#### Scenario: 内部 facade 被删除
- **WHEN** 一个 facade 只 re-export owner 符号、没有独立 behavior、没有 current public import 契约且内部调用方可迁移
- **THEN** 本 change MAY 删除该 facade
- **AND** 内部源码和测试 MUST 改为导入真实 owner module

#### Scenario: 删除后不保留 wrapper
- **WHEN** 内部 import path 被删除或 helper 被合并
- **THEN** 项目 MUST 不新增等价 compatibility wrapper、lazy export 或 package-level barrel 来维持旧路径
- **AND** breaking change MUST 在实现说明中标记为 internal import surface 收缩

### Requirement: Internal __all__ mirrors are not maintained
内部模块 MUST 不维护大型 `__all__` 只为镜像所有可见符号。`__all__` 只允许用于稳定 public facade、明确 plugin/export 边界或避免 wildcard import 的必要模块。

#### Scenario: 删除无 public 契约的 __all__
- **WHEN** 某模块不是 public facade，也没有 docs 推荐 wildcard import
- **THEN** 本 change MAY 删除其 `__all__`
- **AND** 显式 import 调用方 MUST 继续从真实 owner symbol 导入

### Requirement: Reusable helpers stay domain-local
重复 CSV/JSON/path/slug/float 小工具 MUST 优先复用已有 domain owner helper 或建立领域窄 helper。项目 MUST 不新建跨领域大 `utils` 杂物间来容纳少量无 owner 的小函数。

#### Scenario: 重复 helper 收敛
- **WHEN** 两个以上 current owner 需要同一 schema 或 artifact helper
- **THEN** helper MAY 提取到最窄共同领域 owner，例如 diagnostics artifact、runtime output、config parsing 或 dataset contract
- **AND** helper 所在模块 MUST 不引入训练、dataset、模型或重型可视化依赖，除非该领域 owner 已明确需要
