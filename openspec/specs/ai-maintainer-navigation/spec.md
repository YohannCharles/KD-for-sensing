# ai-maintainer-navigation Specification

## Purpose

定义 MMW T2/baseline 与受限 DeepSense6G T2 的维护导航入口，使维护者只加载当前模型、数据、配置、运行和文档约束，而不把已退役路线误作可用工作流。

## Requirements

### Requirement: 导航只指向 current MMW 与 DeepSense6G surface

`AGENTS.md`、`docs/agent_navigation.md` 与维护索引 MUST 将 MMW 的 T2、S1、AMBER-Full、RMBP-MM 以及 DeepSense6G Scene31–34 的受限 T2 四模态主线作为唯一 current 研究 surface，并指向相应 current specs 与 active change。

#### Scenario: 维护者开始非平凡改动

- **WHEN** 维护者打开导航文档
- **THEN** 文档 MUST 给出 MMW 或 DeepSense6G owner、最小验证和本地产物边界
- **AND** 不得把退役路线列为 current 入口

### Requirement: 历史信息集中

导航 MAY 链接 `docs/retired_routes.md`，但 MUST 不复制历史运行命令、YAML 或兼容说明。

#### Scenario: 维护者查询历史路线

- **WHEN** 维护者需要了解已删除能力
- **THEN** 导航 MUST 指向集中历史说明或 OpenSpec archive
- **AND** 不得要求旧实现仍存在
