# Agent Memory Ledger

本 ledger 记录重复 AI 错误的记忆候选。它不是自动规则生成器，不替代 `AGENTS.md`、`docs/agent_navigation.md`、OpenSpec specs、README 或正式 claim 文档。

## 使用规则

- 同类项目规则错误第二次出现时，可以新增一行候选。
- 候选必须由维护者或后续 OpenSpec change 确认后，才可沉淀到 `AGENTS.md`、navigation、scoped context、skill、test 或 OpenSpec artifact。
- hook、agent、dashboard 或脚本不得自动重写 README、OpenSpec current specs、`AGENTS.md`、`docs/result_claims_registry.md`、`docs/experiment_protocols.md` 或其它正式 claim 文档。
- 记忆候选只记录应如何修正行为；不提交 `dataset/`、`outputs/`、`outputs/cache/`、`logs/`、checkpoint、metrics、figures 或报告。

## 字段

| 字段 | 含义 |
| --- | --- |
| 错误模式 | 重复错误的短名称，例如漏用命令环境、误读 active change、把本地产物当源码。 |
| 触发场景 | 该错误通常在什么请求、文件或工作流中出现。 |
| 正确规则 | 应遵守的项目规则，尽量指向现有权威文档。 |
| 建议沉淀位置 | 可能更新的 `AGENTS.md`、navigation、scoped context、skill、test 或 OpenSpec artifact。 |
| 验证命令 | 防止复发的 focused check；Python 命令必须使用 `conda run -n kd_mm_beam ...`。 |
| 人工确认状态 | `candidate`、`confirmed`、`merged` 或 `rejected`。 |

## 候选清单

| 错误模式 | 触发场景 | 正确规则 | 建议沉淀位置 | 验证命令 | 人工确认状态 |
| --- | --- | --- | --- | --- | --- |
| 暂无已确认候选 | 新候选需由维护者补充 | 先读 `AGENTS.md`、`docs/agent_navigation.md` 和 scoped context | 待定 | `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` | candidate |
