# Codex 协作护栏

本文件记录 Codex 任务和人工审查的协作提示。它只是一层本地协作护栏，不替代 OpenSpec、focused tests 或 human review。

## Agent Task Prompt

发给 Codex 前，先补齐这些字段：

- 目标：本次任务要交付什么。
- 非目标：哪些问题本次不处理。
- OpenSpec：active change id；窄修复则说明无需 change 的理由。
- 相关上下文：README、`docs/agent_navigation.md`、scoped context、spec、测试或 PR 链接。
- 允许读取的本地产物：默认无；如需读 `outputs/analysis/...`、`logs/...` 或 checkpoint sidecar，写明只读路径和用途。
- 禁止触碰路径：`dataset/`、`outputs/`、`logs/`、cache、checkpoint、`/root/.container_env`、`/etc/profile`、`/etc/environment`、SSH 配置和系统凭证。
- 禁止操作：不要自动 archive、reset、delete、move outputs、启动真实训练、提交本地数据或恢复退役入口。
- 预期验证：至少列出对应 OpenSpec validate、focused pytest、架构/安全 guard 或 CLI smoke；不能运行时要求说明原因。
- 停止条件：范围漂移、设计冲突、验证红点、需要真实数据/GPU、涉及系统配置或会覆盖用户改动时暂停。

## AI Review Prompt

推荐 review 请求：

```text
请按代码审查姿态检查这个 PR，优先找 regression、missing tests、安全/产物边界、OpenSpec drift、claim caveat 和回滚风险。
不要自动 merge、不要自动升级 claim、不要清理 outputs/logs/cache/checkpoint，也不要建议恢复退役入口。
请明确指出需要补跑的 focused validation，并区分 blocker、warning 和 optional cleanup。
```

Codex review 只能作为附加信号。PR 仍必须通过项目要求的 OpenSpec validate、focused tests 和 human review；claim、paper table 或 dashboard candidate 的状态升级仍按 `docs/result_claims_registry.md` 与人工审阅执行。

## Expected Validation

- OpenSpec change：`openspec validate <change> --strict`
- 全局规格：`openspec validate --all --strict`
- 架构/入口/产物边界：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- CLI/config：`make verify-cli-config`
- compile：`make verify-compile`
- 协作护栏：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

这些检查默认不读取真实 `dataset/`、不启动训练、不加载 checkpoint，也不写入 `outputs/`、`logs/` 或 cache。
