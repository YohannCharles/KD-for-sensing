## MODIFIED Requirements

### Requirement: 当前研究简报
项目 MUST 提供短研究简报，帮助 agent 定位当前主线、冻结方法、退役路线、claim 升级条件和下一步证据缺口。简报 MUST 不依赖 research dashboard/harvester，也 MUST 不替代主线目录、协议表、人工 claim registry、experiment matrix 或 OpenSpec specs。

#### Scenario: 研究简报覆盖当前主线
- **WHEN** AI agent 读取研究简报
- **THEN** 文档 MUST 标明 final C2/U-Mask 主线、主要 controls、MMW/CSI supporting、pending evidence 和不要恢复的路线
- **AND** 文档 MUST 指向人工 claim/protocol owner

#### Scenario: 研究简报不冒充正式 claim
- **WHEN** 简报提到本地结果、mock/smoke 或 pending evidence
- **THEN** 文档 MUST 标明 claim status 或指向 `docs/result_claims_registry.md`
- **AND** 文档 MUST 不依赖 dashboard candidate 作为正式证据

### Requirement: 只读角色 agent 和 skills
项目 MAY 定义只读角色 agent 或 skills，用于 claim audit、experiment triage、surface audit、literature scouting 或其它高噪声分析任务。只读角色 MUST 不修改源码、OpenSpec、README、claim registry、配置、运行产物或 checkpoint，也 MUST 不要求 project surface doctor 产品存在。

#### Scenario: 只读角色返回建议
- **WHEN** 用户或主 agent 调用 claim auditor、experiment triage、surface auditor 或等价角色
- **THEN** 该角色 MUST 只读取允许的 tracked docs/source 或用户明确指定的本地产物
- **AND** 输出 MUST 是建议、风险、缺口或候选任务，不得直接修改文件

#### Scenario: 角色不得绕过项目边界
- **WHEN** 角色 agent 需要运行 Python 检查或引用项目命令
- **THEN** 命令 MUST 使用 `conda run -n kd_mm_beam ...`
- **AND** 角色 MUST 不启动真实训练、清理本地产物、提交 checkpoint、恢复退役入口或绕过 `src/kd_sensing`
