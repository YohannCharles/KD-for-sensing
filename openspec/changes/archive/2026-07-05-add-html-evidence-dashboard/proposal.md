## Why

前一批 6 个 change 已经实现并归档后，20 个优化点中仍缺少一个明确的静态 HTML dashboard/export 能力。当前 `kd-sensing-research-dashboard` 已能输出文本摘要、JSON 和 ledger，但研究者仍需要一个可直接打开、可归档到 ignored output root、能把 run state、claim readiness、paper readiness 和 next action 放在同一页面里的 HTML 证据面板。

## What Changes

- 为现有 research dashboard 增加静态 HTML 输出契约，要求 HTML 与 JSON summary 使用同一数据来源。
- 增加 CLI 输出路径约束，例如 `--output-html` 或等价参数，默认输出到 ignored `outputs/analysis/` 下或用户显式路径。
- HTML 页面必须展示 run 状态、active OpenSpec change、pending/unverified claim、upgradable candidate、paper export gate、缺失 evidence 和 next actions。
- HTML 渲染必须离线可打开，不依赖外部 CDN、远程 JS、训练进程、真实数据读取或 checkpoint 内容。
- 保持只读边界：生成 dashboard 不自动启动训练、清理产物、移动 checkpoint、修改 claim registry 或修改 current docs。
- 增加 focused tests，覆盖 HTML escaping、空输入 fallback、candidate-only 标记、CLI help 和输出文件生成。

## Capabilities

### New Capabilities

- `html-evidence-dashboard`: 定义静态 HTML evidence dashboard 的输入 summary、页面内容、离线渲染、安全边界和输出产物约束。

### Modified Capabilities

- `research-claim-harvester`: 将 daily research dashboard 从文本/JSON 扩展为文本、JSON 和静态 HTML 三种输出。
- `mainline-experiment-documentation`: 明确 paper readiness dashboard 可以产出 HTML report，但不得把 candidate-only 内容写成正式论文结论。
- `project-entrypoint-lifecycle`: 约束 HTML dashboard CLI 仍是只读诊断入口，不新增重复 wrapper 或长期本地脚本入口。

## Impact

- 影响 `src/kd_sensing/diagnostics/research_claim_harvester_dashboard.py` 或新增窄 HTML renderer。
- 影响 `src/kd_sensing/cli/research_dashboard.py` 的参数和输出提示。
- 影响 `pyproject.toml` 中既有 `kd-sensing-research-dashboard` 行为说明，但不需要新增 console script。
- 影响 README、`docs/agent_context/claims.md`、`docs/agent_context/diagnostics.md` 或相关实验文档中的 dashboard 输出说明。
- 影响测试：新增或扩展 `tests/test_research_claim_harvester.py`、`tests/test_cli_help.py` 和 focused CLI/output 测试。
