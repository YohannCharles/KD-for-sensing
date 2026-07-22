# project-entrypoint-lifecycle Specification

## Purpose

定义 current T2/baseline/CMSBL 的三个 public CLI 与最小本地 helper 边界。

## Requirements

### Requirement: Public CLI 仅保留训练、评估和预处理

`pyproject.toml` MUST 只声明 `kd-sensing-train`、`kd-sensing-evaluate` 和 `kd-sensing-preprocess`。它们 MUST 是薄 parser/config glue，并调用包内 owner。

#### Scenario: 安装后检查入口

- **WHEN** 用户检查 console scripts 或运行 help
- **THEN** 只应发现这三个 public command
- **AND** 每个命令 MUST 在 `kd_mm_beam` 环境显示 help

### Requirement: 本地 scripts 保持最小闭包

仓库内 Python script MUST 只保留 MMW all-weather、BPA/CMA、必要 summary 和 compile verification。CMSBL MUST 复用现有 train/evaluate，不新增 wrapper、GPU runner 或 console script。

#### Scenario: 扫描 scripts

- **WHEN** 架构测试枚举 `scripts/*.py`
- **THEN** 每个文件 MUST 能追溯到 current workflow owner
- **AND** retired experiment script MUST 不存在

### Requirement: public CLI 拒绝未知参数和 override

三个 package CLI MUST 拒绝未知 option、以连字符开头的裸 override 和不存在的 dotted config path；只允许显式 `--override key=value` 或已知 `key=value`。

#### Scenario: 用户拼错参数

- **WHEN** 用户传入未知参数或 dotted path
- **THEN** CLI MUST 非零退出并给出错误
- **AND** 不得启动训练或评估

### Requirement: README 声明 MMW generated-config workflow

README 的 MMW training example MUST 使用 retained launcher 生成配置，且不得将 architecture-only T2 YAML 表述为可直接完成 MMW 15-domain training 的命令。

#### Scenario: 用户按 README 启动 MMW

- **WHEN** 用户遵循 MMW workflow
- **THEN** workflow MUST 提供 condition、scene、split 和 profile
