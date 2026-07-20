## ADDED Requirements

### Requirement: evidence fingerprint 必须由实际 payload 验证
generated config、training profile、candidate、probe、checkpoint 和 manifest 的 fingerprint MUST 从规范化的实际 payload 重新计算；记录的 fingerprint 与实际 payload、profile canonical values、training/scheduler fields 任一不一致时 MUST fail closed。

#### Scenario: dry-run 后配置被修改
- **WHEN** `--launch-existing` 读取的 generated YAML 与 manifest 中记录的 fingerprint 不一致
- **THEN** launcher MUST 拒绝启动该 job
- **AND** 不得仅因 YAML 文件存在而继续

### Requirement: launcher 状态写入必须可恢复
screening launcher MUST 在启动子进程前原子写入 planned/running status；启动失败时 MUST 关闭日志并终止本次已启动的 sibling process，留下可诊断 manifest。

#### Scenario: 第二个子进程启动失败
- **WHEN** launcher 在一批 jobs 中遇到 `Popen` 失败
- **THEN** 已启动的 jobs MUST 被终止并记录失败原因
- **AND** manifest 不得保留无法解释的 running 状态
