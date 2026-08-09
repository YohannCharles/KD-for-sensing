# Repository Boundaries Specification

## Purpose

定义 current OpenSpec、公共入口、维护文档和本地产物的最小边界，确保历史实验不会再次作为当前实现约束或运行输入被加载。

## Requirements

### Requirement: current OpenSpec 必须保持显式且隔离

`openspec/specs/` MUST 保留声明中的 current capabilities。完成或停止的 change MUST 从工作树移除；历史由 Git 和仓库外快照追溯，不得保留 `openspec/changes/archive/`。工作树至多保留一个 active research change，且该 change MUST 可单独执行 strict validation。

#### Scenario: 枚举 current OpenSpec

- **WHEN** 维护者查看 `openspec/specs/` 和 `openspec/changes/`
- **THEN** 只能看到声明的 current specs 与至多一个 active research change
- **AND** active change MUST 可单独执行 strict validation

### Requirement: 四模态 topology-probing 只能扩大本地研究面

研究主线 MAY 增加窄模型/loss owner、registry entry、focused tests、`tools/configs/topology_predictor/` 和本地 runner/evaluator。README 和维护导航 MUST 将其描述为 research mainline，同时继续将 U0、AMBER-Full、RMBP-MM、DeepSense6G 与正式 MMW protocol 标记为保留基础。它 MUST 不增加 public CLI、canonical `configs/mmw/` recipe、兼容聚合层或绕过 `src/kd_sensing` 的运行入口。

#### Scenario: 检查 topology-probing source surface

- **WHEN** 维护者审计四模态 topology-probing 变更
- **THEN** public CLI 数量和 canonical MMW recipe MUST 保持不变
- **AND** 本地运行 MUST 通过共享 trainer、config loader、registry 和数据协议

### Requirement: 历史失败分支必须按 owner 成组退出

退出主线的本地实验模型、runner、config、脚本、诊断与测试 MUST 成组移除。clean/block 数据协议、train-only topology likelihood 与正式 split/topology audit MUST 不因源码收敛被删除或迁移。ignored 历史输出只有在用户明确授权、目标目录精确列出且不包含当前保护项时 MAY 删除。

#### Scenario: 收敛失败实验

- **WHEN** 维护者归档历史 change 并删除失败 owner
- **THEN** 当前 source tree MUST 不残留孤立 import、runner、config 或测试
- **AND** topology predictor 与保留 stable baseline 的 focused import/test MUST 继续通过

#### Scenario: 清理历史 ignored outputs

- **WHEN** 用户明确授权删除列名的旧协议、失败路线或动态融合输出目录
- **THEN** 清理前 MUST 验证每个目标位于 ignored `outputs/` 且不属于当前保护清单
- **AND** 正式 split/cache、ULA-DFT topology audit、train-only calibration 与 dataset MUST 保留

### Requirement: 公共入口和维护文档必须最小

项目 MUST 只提供 train、evaluate、preprocess 三个 public CLI。README、配置说明和维护导航 MUST 只描述 four-modal topology predictor + TBCP active research mainline、U0/AMBER-Full/RMBP-MM/DeepSense6G 稳定路线与正式 MMW protocol。

#### Scenario: 查看当前入口

- **WHEN** 用户读取 README 或运行 CLI help
- **THEN** 不得看到已退役路线被描述为 current workflow

### Requirement: 本地产物与 cache 不得被清理触碰

`dataset/`、`outputs/`、`outputs/cache/`、legacy `cache/`、日志和 checkpoint MUST 保持本地边界。源码、current specs、文档和 canonical config MUST 不依赖其内容；trajectory protocol 与 topology predictor 本地产物 MUST 只写入 ignored output/cache 目录。只有用户明确确认的精确旧产物清单 MAY 被删除。

#### Scenario: 收敛仓库表面

- **WHEN** 维护者删除历史 source、OpenSpec 或文档
- **THEN** 未列入用户明确确认清单的本地产物和 cache MUST 不被移动、删除或改写

### Requirement: topology predictor 产物必须保持本地边界

Resolved config、diagnostics、checkpoint 和 smoke 输出 MUST 写入调用方指定的 ignored output 目录，不得作为源码、OpenSpec 证据或 tracked fixture 提交。

#### Scenario: 执行 topology predictor smoke 或评估

- **WHEN** runner 生成任何运行产物
- **THEN** tracked source tree MUST 不出现 checkpoint、日志、cache、resolved config 或评估 JSON
