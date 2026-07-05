## ADDED Requirements

### Requirement: Config list 和 doctor
项目 MUST 提供 config list/doctor 能力，用于按 config family、lifecycle、formal/smoke/local/manual、是否需要真实数据、默认输出边界和 focused validation 对 tracked YAML 与 virtual config route 进行只读分类。Config doctor MUST 不将退役路线恢复为 virtual alias 或实体 YAML。

#### Scenario: 列出当前 config family
- **WHEN** 开发者运行 config list
- **THEN** 输出 MUST 按 canonical root、experiment family、diagnostics、preprocess、local/manual、baseline reproduction 和 retired guard 分类
- **AND** 每条输出 MUST 指向对应 README/docs/OpenSpec 或 inventory 来源

#### Scenario: recipe migration 候选
- **WHEN** config doctor 发现多个实体 YAML 可由同一 recipe 无损生成
- **THEN** doctor MUST 将其标记为 recipe migration candidate
- **AND** 只有在保留 experiment name、objective、dataset split、model/loss/training/output/checkpoint 语义和 focused tests 后，后续 change 才能删除实体 YAML
