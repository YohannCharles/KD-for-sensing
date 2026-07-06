## Why

项目已有本地 OpenSpec 和 focused validation，但缺少面向多人/多 agent 协作的 PR/Issue 原生闭环、分层 CI、自动安全检查和脏工作树收口规则。随着 agentic/vibe-coding 使用增加，最容易失控的不是单次代码生成，而是未验证 PR、重复 CI 红点、秘密/产物误提交、completed change 未归档和 untracked archive 混入当前判断。

## What Changes

- 新增 agentic collaboration guardrails 能力，定义 Issue/PR 模板、agent review prompt、OpenSpec change id、验证命令、产物边界和 claim 状态字段。
- 扩展分层 CI 策略：quick verify 保持轻量；CLI/config、compile、doctor、security/secret scans 可作为 PR、manual、scheduled 或 nightly 层运行。
- 定义安全与依赖护栏：secret scan、系统启动/凭证文件禁止修改、checkpoint/dataset/runtime artifacts 禁止提交、shell runner 危险命令检查和依赖审计。
- 定义脏工作树和 change closeout preflight：报告 active/complete/archive/untracked 状态，不自动清理、不自动 archive、不覆盖用户改动。
- 定义 agent review 集成边界：Codex/GitHub Copilot 或其它 agent 可以做 review 信号和修复建议，但 merge/claim 升级仍需人类和现有验证。

## Capabilities

### New Capabilities
- `agentic-collaboration-guardrails`: 定义 agentic 协作中的 PR/Issue、CI、安全扫描、脏工作树收口和 review 集成边界。

### Modified Capabilities

无。该能力会引用项目健康护栏、入口生命周期、AI navigation 和 OpenSpec 文档健康，但本 change 先独立定义协作治理契约。

## Impact

- 可能新增 `.github/ISSUE_TEMPLATE/*`、`.github/pull_request_template.md`、GitHub Actions workflow、security scan 脚本、preflight/doctor scope 或文档。
- 可能扩展 `kd-sensing-project-surface-doctor`、`tests/test_architecture_boundaries.py` 或新增 focused tests。
- 可能更新 README、AGENTS、agent navigation、inventory 和 docs/agent_context。
- 不自动修改本地数据、outputs、logs、cache、checkpoint，不恢复退役入口，不改变训练/评估 runtime 语义。
