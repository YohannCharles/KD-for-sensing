# OpenSpec 任务上下文

用于 proposal、design、spec、tasks、apply、archive、complete active change 收口和文档健康治理。

## 先读

- `AGENTS.md` 的 OpenSpec 和命令环境规则
- `docs/agent_navigation.md` 的当前状态检查顺序
- 目标 change 的 `openspec/changes/<change>/proposal.md`、`design.md`、`tasks.md` 和 `specs/**/*.md`
- 当前 specs：`openspec/specs/`

## 状态判断

1. `openspec list --json` 查看 active changes。
2. `openspec status --change <change> --json` 判断 schema、artifact 状态和 complete 状态。
3. `openspec instructions apply --change <change> --json` 获取 contextFiles、任务进度和动态指令。
4. complete 但未 archive 的 change 是治理收口项，不是新的实施范围，除非用户明确要求继续。

## 边界

- 非平凡功能、架构、训练流程、数据契约、配置兼容或公共入口变化先走 OpenSpec change。
- 实现中发现范围变化，先更新对应 OpenSpec artifact，再继续落代码。
- Archive 是历史记录；归档内容不能覆盖当前 specs、inventory 或 README。
- OpenSpec 文档不要复制 README quickstart 或完整源码清单。

## 验证

- `openspec validate <change> --strict`
- `openspec validate --all --strict`
- `openspec status --change <change>`
