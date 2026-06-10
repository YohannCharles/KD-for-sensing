## Why

当前项目支持面已经出现多处可审计漂移：架构边界测试暴露 `configs/fusion/` 根目录数量超过约定、OpenSpec 当前 spec 仍有脚手架 `TBD` Purpose，脚本和 inventory 中也存在过期命名与统计。现在需要先用一个明确 change 收敛这些漂移，避免后续删除冗余时误删当前主线入口或继续维护已经过期的实验表面。

## What Changes

- 修复当前架构边界红点，使项目表面积 inventory、OpenSpec spec 文本和测试 guardrail 与真实仓库状态一致。
- 收缩 `configs/fusion/` 根目录支持面：保留长期 canonical 配置，将实验特化、临时复现或重复低内存配置归档、迁移或删除到明确位置，并更新对应文档。
- 修复 shell orchestration、README/文档和 OpenSpec 中已发现的过期引用，例如已不存在的 hardening matrix 配置名和脚手架 Purpose。
- 复核无调用或孤立的源码候选，只删除不属于公共 API、无当前入口依赖且有测试覆盖的冗余实现；对兼容导入路径和当前文档声明的入口保持保守。
- 清理本地忽略产物的建议边界：允许删除 `__pycache__`、`.pytest_cache`、egg-info 和明确备份/临时产物；`outputs/`、`logs/`、cache 等实验产物继续遵循 manifest 或用户显式确认。
- 不引入新的训练流程、模型能力、数据契约或外部依赖。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`: 增加当前支持面漂移修复、配置根目录收缩、脚本文档一致性和本地产物清理边界的需求。
- `project-architecture`: 增加 OpenSpec/current inventory hygiene 与架构 guardrail 必须反映真实项目表面的要求。

## Impact

- 影响文档与规范：`openspec/specs/` 中相关 capability、`docs/project_surface_inventory.md`、可能涉及 README/复现实验说明。
- 影响配置：`configs/fusion/` 根目录的 YAML 数量、分类和引用路径。
- 影响脚本：已过期的 shell orchestration 配置引用。
- 影响测试：`tests/test_architecture_boundaries.py` 以及必要的窄向回归检查。
- 不应影响数据集目录、历史权重、当前训练输出或本地实验产物；这些产物不得被自动纳入源码变更。
