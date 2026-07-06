# Read-Only Agent Roles

本文件定义可选的只读分析角色。它们帮助主 agent 降低高噪声调查成本，但不直接修改文件、不启动训练、不清理产物、不 archive OpenSpec change，也不绕过主 agent 的 OpenSpec 边界。

## Discovery

主 agent 可以在用户明确要求审计、triage、doctor review 或 literature scout 时临时采用这些角色说明。若未来把角色做成工具专属 agent、skill 或 subagent 文件，必须继续引用本文件、`AGENTS.md`、`docs/agent_navigation.md` 和对应 OpenSpec capability。

所有角色的 Python 检查命令必须使用 `conda run -n kd_mm_beam ...`。默认只读取 tracked docs/source/OpenSpec；只有用户明确提供本地产物路径时，才可只读查看对应 `outputs/`、`logs/` 或 report。

## Roles

| Role | Trigger | Read Scope | Output |
| --- | --- | --- | --- |
| `claim-auditor` | 审核 claim status、provenance、paper table readiness 或 pending candidate | `docs/result_claims_registry.md`、`docs/experiment_protocols.md`、`docs/mainline_model_catalog.md`、用户指定 summary | 建议、风险、缺失字段、候选 next action |
| `experiment-triage` | 汇总 run state、seed 覆盖、fresh eval 缺口、budget 风险 | README、`docs/experiment_matrix.md`、相关 manifest、用户指定 run summary | 缺口列表、优先级、不可比原因 |
| `surface-doctor-reviewer` | 阅读 project surface doctor 输出或 guardrail warning | doctor 输出、`docs/project_surface_inventory.md`、`docs/agent_navigation.md`、OpenSpec specs | inventory/guardrail 收口建议 |
| `literature-scout` | 对比外部论文、官方 artifact 状态和本仓库 baseline 对应关系 | `docs/literature_matrix.md`、用户提供论文摘要或链接、相关 claim docs | 文献差距、artifact caveat、待补 BibTeX 候选 |

## Forbidden Actions

- 不直接写 README、OpenSpec、`AGENTS.md`、claim registry、配置、源码、checkpoint 或运行产物。
- 不启动真实训练、长跑评估、清理命令、删除命令、archive 命令或 git 操作。
- 不把 pending、mock/smoke、upper-bound、historical、blocked 或 not-comparable evidence 升级成正式 claim。
- 不恢复旧入口、兼容 facade、退役实体 YAML 或绕过 `src/kd_sensing` 包结构。

## Handoff

角色输出只能作为建议交给主 agent。真正落地代码、文档、OpenSpec artifact 或 claim 更新时，主 agent 必须重新确认当前 change 范围、文档边界、产物边界和 focused validation。
