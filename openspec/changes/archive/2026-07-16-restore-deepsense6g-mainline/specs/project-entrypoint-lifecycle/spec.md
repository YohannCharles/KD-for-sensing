## MODIFIED Requirements

### Requirement: Public CLI 仅保留训练、评估和预处理

`pyproject.toml` MUST 只声明 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。它们 MUST 是薄 parser/config glue，并调用包内 owner；训练和评估入口 MUST 能消费 current MMW 或 DeepSense6G canonical config，且不得因支持 DeepSense6G 新增旧入口。

#### Scenario: 安装后检查入口

- **WHEN** 用户检查 package console scripts 或运行 help
- **THEN** 只应发现这三个 public command
- **AND** 每个命令 MUST 在 `kd_mm_beam` 环境中正常显示 help
