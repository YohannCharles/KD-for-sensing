## MODIFIED Requirements

### Requirement: 公共入口和维护文档必须最小

项目 MUST 只提供 train、evaluate、preprocess 三个 public CLI。README、配置说明和维护导航 MUST 只描述 Clean MMW U0、AMBER-Full、RMBP-MM、DeepSense6G 和正式 MMW protocol 工作流。trajectory-disjoint 的构建、M0--M4 与监控 MUST 保持为 `tools/` 和 `scripts/` 下的本地研究入口，不得扩展 public CLI 或 canonical recipe。

#### Scenario: 查看当前入口

- **WHEN** 用户读取 README 或运行 CLI help
- **THEN** 不得看到已退役路线被描述为 current workflow
- **AND** public CLI 数量 MUST 保持为三个

### Requirement: 本地产物与 cache 不得被清理触碰

`dataset/`、`outputs/`、`outputs/cache/`、legacy `cache/`、日志和 checkpoint MUST 保持本地边界。源码、current specs、文档和 canonical config MUST 不依赖其内容；trajectory protocol 生成物和 M0--M4 运行物 MUST 只写入 `outputs/mmw_trajectory_split/` 且不得进入源码变更。

#### Scenario: 实现或收敛 trajectory protocol

- **WHEN** 维护者修改协议源码、OpenSpec、测试或文档
- **THEN** 既有本地产物和 cache MUST 不被移动、删除或纳入 Git
