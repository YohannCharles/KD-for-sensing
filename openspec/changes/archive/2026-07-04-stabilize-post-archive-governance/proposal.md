## Why

近期完成多轮表面积收口、OpenSpec 归档和热点拆分后，项目整体验证已经稳定，但仍存在少量“归档后治理漂移”：维护索引中保留了已不可运行的 change-specific OpenSpec 校验命令，inventory 的统计口径没有明确区分 tracked-only 与当前工作树，部分拆分后的普通测试仍局部插入 `tests/` 路径，且归档目录与删除 active change 的提交完整性需要更明确的 guardrail。

本 change 旨在用一个低风险、无运行时副作用的治理收口，把这些红点固化为文档、OpenSpec 和架构边界测试规则，避免后续维护者把 archive、generated YAML、陈旧命令或测试 bootstrap 误读为当前支持面。

## What Changes

- 更新维护上下文索引的验证命令规则，禁止 focused validation 长期引用 `openspec list --json` 中不存在的 active change；归档后默认使用 `openspec validate --all --strict` 加 focused pytest。
- 明确 project surface inventory 中源码/测试/脚本/config 数量基线的统计口径，区分当前工作树 on-disk 扫描、tracked-only 扫描和历史 CodeGraph/AST 基线，避免把约数当作硬预算。
- 收紧普通 pytest helper 导入规范：拆分后的测试 helper 应通过 shared pytest bootstrap 可解析的包路径导入，不在普通测试文件头部维护 `tests/` 路径注入片段。
- 增加归档后版本控制完整性检查：当 active change 目录删除并出现对应 `openspec/changes/archive/<date>-<change>/` 时，提交前必须把删除和 archive 新增作为成对状态审计。
- 可选消除 MMW helper 中 pandas fragmentation warning，作为无语义变更的性能/噪声清理；该项不改变 dataset sample、metadata、label 或训练语义。
- 不新增训练、评估、预处理、诊断 CLI；不读取真实 `dataset/`，不写入 `outputs/`、`logs/`、cache、checkpoint 或本地运行产物。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `maintainer-context-index`: 维护索引的验证命令必须引用当前可运行的 OpenSpec/spec/pytest 检查，并且统计基线必须声明来源、范围和口径。
- `project-health-guardrails`: 架构边界测试需要覆盖陈旧 OpenSpec validation 命令、普通测试局部 `tests/` path bootstrap、inventory 统计口径和无副作用 focused validation。
- `spec-lifecycle-boundaries`: 归档后状态收口需要明确 active change 删除与 archive 新增的成对提交审计，防止把未跟踪 archive 或删除目录误读为 active requirement。

## Impact

- 文档与治理数据：`docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`、`docs/agent_navigation.md` 或相关 README 段落。
- OpenSpec：修改上述 3 个 current specs 的 delta，并保留此 change 的 proposal/design/tasks。
- 测试：`tests/test_architecture_boundaries.py` 以及使用共享 helper 的普通测试文件。
- 可选源码清理：`src/kd_sensing/data/datasets/mmw_columns.py` 中的 DataFrame 列构造方式。
- 验证：`openspec validate --all --strict`、架构边界测试、Scene31 focused tests、拆分后 helper 测试、MMW preparation focused test 和最终全量 pytest。
