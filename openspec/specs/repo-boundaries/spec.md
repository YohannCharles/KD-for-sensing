# Repository Boundaries Specification

## Purpose

定义 current OpenSpec、公共入口、维护文档和本地产物的最小边界，确保历史实验不会再次作为当前实现约束或运行输入被加载。

## Requirements

### Requirement: current OpenSpec 必须保持最小

`openspec/specs/` MUST 只保留 `clean-data-integrity`、`mmw-trajectory-disjoint-protocol`、`u0-mainline` 与 `repo-boundaries`。完成或停止的 change MUST 从工作树移除；历史由 Git 和仓库外快照追溯，不得保留 `openspec/changes/archive/`。PCPF-T 实施期间 MUST 是唯一 active change。

#### Scenario: 枚举 current OpenSpec

- **WHEN** 维护者查看 `openspec/specs/` 和 `openspec/changes/`
- **THEN** 只能看到四个 current spec 和 `add-pcpf-temporal-risk-fusion` 这一个 active change

### Requirement: 公共入口和维护文档必须最小

项目 MUST 只提供 train、evaluate、preprocess 三个 public CLI。README、配置说明和维护导航 MUST 描述 PCPF-T active research mainline、U0/AMBER-Full/RMBP-MM/DeepSense6G 稳定路线、正式 MMW protocol，以及隔离保留的 CSI/TSPC 本地研究面。

#### Scenario: 查看当前入口

- **WHEN** 用户读取 README 或运行 CLI help
- **THEN** 不得看到已退役路线被描述为 current workflow

### Requirement: 本地产物与 cache 不得被清理触碰

`dataset/`、`outputs/`、`outputs/cache/`、legacy `cache/`、日志和 checkpoint MUST 保持本地边界。源码、current specs、文档和 canonical config MUST 不依赖其内容；trajectory、CSI/TSPC 与 PCPF-T 本地运行产物 MUST 只写入 ignored output/cache 目录。

#### Scenario: 收敛仓库表面

- **WHEN** 维护者删除历史 source、OpenSpec 或文档
- **THEN** 本地产物和 cache MUST 不被读取、移动、删除或改写
