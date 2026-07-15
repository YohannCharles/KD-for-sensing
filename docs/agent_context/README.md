# Agent Context 路由

本目录提供按任务渐进加载的维护上下文。它不是需求契约，也不是源码目录镜像；权威仍是 `AGENTS.md`、README、`openspec/specs/`、active OpenSpec change 和 `docs/project_surface_inventory.md`。

## 使用顺序

1. 先读 `AGENTS.md`、`docs/agent_navigation.md` 和 `docs/maintainer_context_index.yaml`，确认命令环境、任务路由和本地产物边界。
2. 按任务只读取下表中匹配的 scoped context。
3. 再读取该 context 指向的 OpenSpec specs、inventory 段落、owner module 或 focused tests。
4. 修改后运行 context 中列出的最小验证命令；无法运行时在最终说明中记录原因。

## 路由表

| Route id | Scoped context | 适用任务 |
| --- | --- | --- |
| `model` | `docs/agent_context/models.md` | 模型、forward、registry、baseline、representation core 或组件扩展 |
| `data` | `docs/agent_context/data.md` | dataset、batch contract、modality profile、split 或本地数据边界 |
| `config` | `docs/agent_context/configs.md` | YAML、virtual config、canonical recipe、迁移 guard 或配置解析 |
| `cli` | `docs/agent_context/cli.md` | console scripts、包内 CLI、`scripts/` 本地 runner 或 help smoke |
| `diagnostics` | `docs/agent_context/diagnostics.md` | run index、U-Mask eval matrix、runtime cleanup、paper export 或 MMW diagnostics |
| `openspec` | `docs/agent_context/openspec.md` | proposal、spec、tasks、archive、complete active change 收口 |
| `documentation` | `docs/agent_context/documentation.md` | README、AGENTS、docs lifecycle、inventory、导航和文档健康 |
| `claims` | `docs/agent_context/claims.md` | result claim registry、论文表格、provenance、claim gate 或本地结果入账 |
| `atlas` | `docs/agent_context/atlas.md` | 需要快速扫视 spec/config/claim owner、lifecycle、validation 和 caveat |

## Complete active change

如果 `openspec list --json` 或 `openspec status --change <name>` 显示某个 active change 已 complete 但尚未归档，应把它当作治理收口项：先确认工作树和验证状态，再 archive 或记录 deferral。除非用户明确要求继续该 change，不要把 complete active change 当作新的实施范围。
