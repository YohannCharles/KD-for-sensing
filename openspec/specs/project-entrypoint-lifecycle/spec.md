# project-entrypoint-lifecycle Specification

## Purpose

定义 MMW T2/baseline 与受限 DeepSense6G T2 的 public CLI 与本地 helper 边界，防止研究脚本、结果汇总或历史工具重新扩展为额外 package console command。
## Requirements
### Requirement: Public CLI 仅保留训练、评估和预处理

`pyproject.toml` MUST 只声明 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。它们 MUST 是薄 parser/config glue，并调用包内 owner；训练和评估入口 MUST 能消费 current MMW 或 DeepSense6G canonical config，且不得因支持 DeepSense6G 新增旧入口。

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
