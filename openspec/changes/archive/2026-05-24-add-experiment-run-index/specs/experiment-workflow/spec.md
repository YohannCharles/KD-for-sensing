## ADDED Requirements

### Requirement: 运行状态产物
训练和评估入口 MUST 尽量写出机器可读运行状态产物，使 run index 能判断启动、正常完成和 Python 异常失败。状态产物 MUST 保持轻量，并且 MUST 不改变现有 `final_config.yaml`、`resolved_config.yaml`、`metrics.json`、`train_log.json`、checkpoint 或 TensorBoard 语义。

#### Scenario: 训练启动写出状态
- **WHEN** 训练入口创建 run_dir 并完成初始配置解析
- **THEN** 系统 MUST 写出 `run_status.json` 或等价 runtime status 字段
- **AND** 状态 MUST 至少包含 `state: running`、run_dir、config path、start time、pid、experiment name、task、objective 和 enabled modalities

#### Scenario: 训练正常完成更新状态
- **WHEN** 训练完成并写出最终 metrics、train log 和 checkpoint metadata
- **THEN** 系统 MUST 将运行状态更新为 `complete`
- **AND** 状态 MUST 记录 end time、duration、primary metric、best checkpoint 和 metrics path

#### Scenario: Python 异常失败更新状态
- **WHEN** 训练或评估入口捕获到未处理 Python exception 并准备退出
- **THEN** 系统 SHOULD 将运行状态更新为 `failed`
- **AND** 状态 SHOULD 记录异常类型、异常消息和可查看的日志路径

#### Scenario: SIGKILL 无法捕获
- **WHEN** 训练进程被系统或用户以不可捕获方式终止
- **THEN** 系统 MAY 无法更新运行状态产物
- **AND** run index MUST 仍能通过日志和 partial artifacts 推断 killed、stale 或 partial 状态
