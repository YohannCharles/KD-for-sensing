## ADDED Requirements

### Requirement: 导航文档提供当前项目一屏摘要
AI 维护导航文档 SHALL 在详细阅读顺序之前提供一段短摘要，用于快速说明当前主线、推荐入口、退役边界、必读文件和最小验证命令。该摘要 MUST 保持为导航入口，不得替代 README quickstart、OpenSpec requirements、project surface inventory 或任务路由表。

#### Scenario: AI 快速判断当前主线
- **WHEN** AI agent 或维护者打开 `docs/agent_navigation.md`
- **THEN** 文档 MUST 在顶部附近说明当前主线和当前推荐入口类别
- **AND** 文档 MUST 同时提醒退役路线不能恢复为当前入口

#### Scenario: 摘要不复制完整治理表
- **WHEN** 项目更新 AI 导航摘要
- **THEN** 摘要 MUST 指向 README、project surface inventory、maintainer context index 和 OpenSpec specs
- **AND** 摘要 MUST NOT 维护完整源码目录清单、完整脚本 allowlist 或完整热点预算表
