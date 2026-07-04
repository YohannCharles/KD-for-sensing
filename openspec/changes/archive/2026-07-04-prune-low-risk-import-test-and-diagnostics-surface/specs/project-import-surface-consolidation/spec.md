## ADDED Requirements

### Requirement: 低风险 import/export 噪音必须可机械清理
当 obsolete future annotations imports、runtime star imports 和 internal-only `__all__` mirrors 没有 current public contract 价值时，项目 MUST 允许删除它们。

#### Scenario: 星号导入被替换
- **WHEN** runtime source uses `from <module> import *` outside a documented public facade
- **THEN** 实现 MUST 将其替换为显式 import，或删除对应 compatibility module
- **AND** architecture tests MUST 继续拒绝新的 runtime star import

#### Scenario: 内部 `__all__` 有理由才保留
- **WHEN** an internal owner module has `__all__`
- **THEN** 它 MUST 代表 current public/export boundary，否则必须在 cleanup 中删除
