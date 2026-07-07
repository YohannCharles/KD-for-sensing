## ADDED Requirements

### Requirement: MMW Town GPS v2 run/plot/compare 可由单一 owner CLI 覆盖
MMW Town GPS v2 的 runner、plotter 和 comparator MAY 收敛为单一 package CLI 或单一 owner module 下的 modes。迁移 MUST 保持旧 plot/compare 行为的输入 artifact、输出文件、metric 字段、排序和 help 语义，除非 current spec 同步修改。

#### Scenario: plot mode 替代旧 plot CLI
- **WHEN** 协作者需要绘制 MMW Town GPS v2 结果
- **THEN** 推荐入口 MAY 是 MMW Town GPS v2 owner CLI 的 `plot` mode 或等价 flag
- **AND** 项目 SHOULD 不保留只转发到 plot helper 的独立 plot console script

#### Scenario: compare mode 替代旧 compare CLI
- **WHEN** 协作者需要比较 MMW Town GPS v2 runs
- **THEN** 推荐入口 MAY 是 MMW Town GPS v2 owner CLI 的 `compare` mode 或等价 flag
- **AND** compare 输出的 metric names、method labels 和排序 MUST 保持可对照

### Requirement: MMW Town wrapper 删除必须更新 console scripts
删除 MMW Town GPS v2 plot/compare wrapper 时，`pyproject.toml` console scripts、CLI help tests、docs 和 inventory MUST 同步迁移到 owner CLI。

#### Scenario: console script 不保留旧别名
- **WHEN** plot/compare 独立入口被 consolidated owner 覆盖
- **THEN** 旧 console script name MUST 从 current package entrypoints 中移除
- **AND** 项目 MUST 不新增等价 legacy alias
