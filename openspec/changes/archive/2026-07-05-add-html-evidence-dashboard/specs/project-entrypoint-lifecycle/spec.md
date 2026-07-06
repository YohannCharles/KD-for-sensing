## ADDED Requirements

### Requirement: HTML dashboard 入口保持只读诊断
HTML evidence dashboard MUST 通过现有 research dashboard CLI 或等价包内 CLI 暴露，并保持只读诊断入口。项目 MUST 不为该 HTML 输出新增重复 wrapper、长期本地 shell 入口、Web 服务入口或绕过 `src/kd_sensing` 包结构的脚本。

#### Scenario: CLI 入口不膨胀
- **WHEN** 实现 HTML dashboard 输出
- **THEN** 项目 MUST 优先扩展 `kd-sensing-research-dashboard` 或其包内 owner
- **AND** `pyproject.toml` MUST 不新增与同一功能重复的 console script
- **AND** README MUST 不把本地 shell wrapper 描述为推荐入口

#### Scenario: HTML dashboard 不启动服务
- **WHEN** 用户请求生成 HTML dashboard
- **THEN** 命令 MUST 生成静态文件并退出
- **AND** 命令 MUST 不启动常驻 Web server、训练队列、清理任务或后台进程
