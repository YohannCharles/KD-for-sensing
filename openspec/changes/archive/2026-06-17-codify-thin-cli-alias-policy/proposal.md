## Why

项目当前保留了 package CLI、`scripts/` 薄 alias、研究诊断脚本和数据准备脚本。薄 alias 本身是合理的，但如果没有明确策略，后续新增入口容易把 parser、业务逻辑、训练 orchestration 和兼容包装混在一起，形成第二套实现或恢复旧入口。

## What Changes

- 明确 CLI/脚本入口职责：package CLI 只承担 parser、参数覆盖、轻量 IO 和调用包内实现；真实 workflow 逻辑必须位于 `baselines/`、`diagnostics/`、`engine/`、`data/` 或对应职责模块。
- `scripts/` 下长期保留文件必须属于 thin alias、research diagnostic、dataset preparation 或 shell orchestration，并在 maintainer context index 中登记 owner module 和 output boundary。
- 新增 CLI 或脚本时必须同步 `pyproject.toml`、maintainer context index、project surface inventory 和架构边界测试。
- 架构测试增加轻量检查，防止 CLI 文件实现大段训练/评估/数据处理逻辑。
- 不删除当前入口，不改变 CLI 参数行为，不恢复 retired route。

## Capabilities

### New Capabilities

### Modified Capabilities

- `project-architecture`: 明确 package CLI、scripts thin alias 和实现模块之间的职责边界。
- `project-health-guardrails`: 健康护栏需要检查 CLI alias 不变厚、不复制 workflow 逻辑，并与 maintainer context index 同步。
- `maintainer-context-index`: entrypoint governance 需要记录 owner module、allowed responsibilities 和 output boundary。

## Impact

- 影响 `docs/maintainer_context_index.yaml` entrypoint schema、`docs/project_surface_inventory.md` 入口说明、`tests/test_architecture_boundaries.py`。
- 可能影响少量 CLI 文件注释或结构，但不改变命令行为。
- 后续新增入口会有更明确的登记和测试要求。
