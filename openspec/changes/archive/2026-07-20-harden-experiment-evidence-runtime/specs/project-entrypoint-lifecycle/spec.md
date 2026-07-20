## ADDED Requirements

### Requirement: public CLI 必须拒绝未知参数和未知 override
三个 current package CLI MUST 拒绝未知 option、以连字符开头的裸 override 和不存在的 dotted config path；只允许显式 `--override key=value` 或不带 option 前缀的已知 `key=value`。

#### Scenario: 用户拼错训练参数
- **WHEN** 用户传入 `--num-wokers` 或 `training.lrr=...`
- **THEN** CLI MUST 以非零状态和可操作错误退出
- **AND** 不得静默启动训练或评估

### Requirement: 文档必须声明 MMW generated-config workflow
README 的 MMW training example MUST 使用 retained launcher 生成的配置或明确说明所需的 domain inventory；它不得将 architecture-only tracked T2 YAML 表述为可直接训练的 MMW command。

#### Scenario: 维护者按 README 启动 MMW
- **WHEN** 用户遵循 README 的 MMW workflow
- **THEN** 该 workflow MUST 提供 condition、scene、split 和 profile 所需的 generated configuration
- **AND** H4/H0 protocol 不得被静默猜测
