## ADDED Requirements

### Requirement: CLI 与实现模块职责分离
项目 SHALL 保持 CLI/脚本入口与真实 workflow 实现的职责分离。Package CLI 和 thin alias MUST 只承担参数解析、配置覆盖、轻量 IO、调用包内实现和 user-facing exit code；训练、评估、benchmark、dataset preparation 或诊断主逻辑 MUST 位于对应职责模块。

#### Scenario: package CLI 调用 owner module
- **WHEN** 新增或修改 package console script
- **THEN** CLI 文件 MUST 调用 `baselines/`、`diagnostics/`、`engine/`、`data/` 或其它对应 owner module 中的实现
- **AND** CLI 文件 MUST 不复制通用训练循环、评估循环、模型 forward 分支或 dataset parsing 主逻辑

#### Scenario: scripts thin alias 不恢复旧入口
- **WHEN** 新增或保留 `scripts/` 下的 thin alias
- **THEN** 脚本 MUST 委托 package CLI 或包内 owner module
- **AND** 脚本 MUST 不恢复 retired route、旧兼容聚合层或仓库根旧式入口

### Requirement: 入口输出边界显式
每个长期保留 CLI 或脚本入口 SHALL 有明确输出边界。入口 MUST 将训练、诊断、cache、checkpoint 和报告输出限定在 ignored 本地产物目录或显式用户指定目录，不得写入源码目录。

#### Scenario: 新诊断入口声明输出边界
- **WHEN** 新增 research diagnostic 或 benchmark CLI
- **THEN** maintainer context index 或 inventory MUST 记录默认输出目录和是否只读
- **AND** 输出 MUST 位于 `outputs/`、`logs/`、dataset preparation target 或显式本地路径边界内
