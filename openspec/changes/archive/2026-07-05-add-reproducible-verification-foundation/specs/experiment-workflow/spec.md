## ADDED Requirements

### Requirement: Verify workflow 不等同训练 workflow
项目的 verify、CI、lint 和 smoke workflow MUST 与真实训练/评估 workflow 保持边界清晰。Verify workflow 只能检查源码、配置、OpenSpec、CLI help、synthetic forward 或 mock schema；真实训练、长时间评估、feature cache 生成和 checkpoint 写入仍 MUST 通过显式训练/评估入口触发。

#### Scenario: CI 不启动真实训练
- **WHEN** CI 或 quick verify 在无真实数据环境中运行
- **THEN** 系统 MUST 不调用长时间 `kd-sensing-train` 真实训练
- **AND** 如需训练路径 smoke，MUST 使用 synthetic/mock fixture 或已有 focused test

#### Scenario: 训练仍使用 package CLI
- **WHEN** 用户需要运行真实实验
- **THEN** 文档 MUST 继续指向 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 或已登记 package CLI
- **AND** verify 入口 MUST 不成为新的长期训练入口
