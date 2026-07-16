# project-entrypoint-lifecycle Specification

## Purpose

定义 MMW T2/baseline 的 public CLI 与本地 helper 边界，防止研究脚本、结果汇总或历史工具重新扩展为额外 package console command。

## Requirements

### Requirement: Public CLI 仅保留训练、评估和预处理

`pyproject.toml` MUST 只声明 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。它们 MUST 是薄 parser/config glue，并调用包内 owner。

#### Scenario: 安装后检查入口

- **WHEN** 用户检查 package console scripts 或运行 help
- **THEN** 只应发现这三个 public command
- **AND** 每个命令 MUST 在 `kd_mm_beam` 环境中正常显示 help

### Requirement: MMW evidence scripts 是 local/manual helper

保留的 all-weather、screening、BPA/CMA 与 summary scripts MUST 有 MMW owner 和 output 边界，但 MUST 不注册为额外 console script。

#### Scenario: 运行本地 helper

- **WHEN** 维护者运行 retained script
- **THEN** 其生成物 MUST 写入 local output root
- **AND** script MUST 不恢复历史 CLI 或 thin alias
