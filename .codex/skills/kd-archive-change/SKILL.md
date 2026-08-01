---
name: kd-archive-change
description: 收尾并移除已完成的 KD-for-sensing OpenSpec change，同时同步 current specs、验证源码与产物边界。用于用户明确要求归档、关闭或 finalize 一个 change；本仓库不创建 `openspec/changes/archive/`，历史仅由 Git 或仓库外快照保留。
---

# KD OpenSpec 收尾

本技能覆盖通用 OpenSpec archive 行为。不要运行会创建本地 archive 目录的标准归档流程。

## 上下文

1. 读取 `AGENTS.md`、`docs/agent_navigation.md`、`docs/maintainer_context_index.yaml`、`docs/agent_context/README.md` 和 `openspec/specs/repo-boundaries/spec.md`。
2. 运行 `openspec list --json` 与 `openspec status --change <change> --json`。
3. 完整读取 change 的 proposal、design、tasks 和全部 delta specs。

## 工作流

1. 仅处理用户明确点名或上下文唯一确定的 change。若名称有歧义，先询问。
2. 检查所有 artifact 和任务确实完成，并确认不存在尚未验证的 claim、协议绑定或实现待办。未完成时继续实施或记录明确 deferral，不得归档。
3. 运行 `openspec validate <change> --strict` 及 change/scoped context 指定的聚焦验证。
4. 将 delta specs 逐项同步到 `openspec/specs/` 的 current specs，并同步受影响的 README、导航和机器可读治理表。不要依赖 generated local artifacts 完成同步。
5. 再次验证 current specs 和实现。确认 `dataset/`、`outputs/`、`logs/`、cache、TensorBoard 与 checkpoint 未进入源码变更。
6. 从工作树移除已同步的 active change。不得创建或保留 `openspec/changes/archive/`；需要额外历史副本时只使用用户指定的仓库外路径。
7. 运行 `openspec validate --all --strict`、`openspec list --json` 和至少 `make verify-quick`，确认 active change、current spec 与架构边界一致。

所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。最终报告同步了哪些 current specs、移除了哪个 active change、运行了哪些验证，以及历史由 Git 或哪个明确的外部快照保留。
