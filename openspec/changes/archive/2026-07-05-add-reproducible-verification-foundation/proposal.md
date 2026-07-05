## Why

项目已有严格的 OpenSpec 和 focused tests，但缺少一个机器可复制的最小验证入口、CI 基线和 tracked 环境描述。对 agentic/vibe-coding 工作流来说，这会让每次自动修复都依赖人工记忆命令和本机环境状态。

## What Changes

- 增加最小 `verify`/CI 基线，覆盖 OpenSpec strict、架构边界、CLI help、配置 characterization 和关键 synthetic tests。
- 增加 tracked 的最小环境声明或导出流程，区分 CPU/smoke 环境和 GPU/训练环境。
- 增加轻量 lint/compile 层，优先覆盖脚本语法、OpenSpec 文档引用和无副作用 smoke。
- 保持真实训练、真实数据、checkpoint、cache 和日志在 ignored 本地产物边界内。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-health-guardrails`: 增加可复制 verify/CI、环境声明和轻量 lint 的健康护栏要求。
- `experiment-workflow`: 明确 verify/CI 不得启动真实训练或读取真实 dataset，训练/评估仍通过现有 CLI。

## Impact

- 可能新增 `Makefile`、`justfile`、`scripts/verify_*.py`、`.github/workflows/*.yml` 或环境文件，具体由实现阶段选择。
- 影响 README、AGENTS、docs/agent_navigation.md 和 inventory 中的验证命令说明。
- 不新增长期训练入口，不改变现有 console scripts。
