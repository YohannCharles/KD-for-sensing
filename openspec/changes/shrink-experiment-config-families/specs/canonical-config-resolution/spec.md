## ADDED Requirements

### Requirement: 实验配置族收缩必须有等价门
删除或生成化 tracked experiment YAML 前，项目 MUST 证明该配置不属于 current canonical、paper/workflow reproduction、diagnostics manifest、claim/evidence input 或必要 focused test fixture；若由 generator/manifest/base config 替代，MUST 证明关键 resolved semantics 等价。

#### Scenario: 删除前检查 current 依赖
- **WHEN** implementation 准备删除 `configs/scene31/`、`configs/fusion/experiments/rbma_missing_workflow*` 或 `configs/fusion/experiments/jepa_image_gps/` 下的实体 YAML
- **THEN** implementation MUST 检查 README、docs、OpenSpec current specs、tests、scripts、claim provenance 和 config loader 引用
- **AND** 仍被 current evidence 或 reproduction 消费的 YAML MUST 保留或先提供等价 generator/manifest 输入

#### Scenario: generator 等价验证
- **WHEN** 实体 YAML 被 generator、template、manifest 或 recipe 替代
- **THEN** focused tests MUST 校验 run name、seed、epoch、sampler、loss weights、missing pattern、dataset split、model primary、output boundary 和关键 overrides
- **AND** 允许差异 MUST 仅限 run identity、输出目录、timestamp 或其它明确非行为字段

#### Scenario: 删除不恢复 retired route
- **WHEN** config family shrink 删除历史或重复 YAML
- **THEN** virtual config、generator 或 migration guard MUST 不把 retired KD、BGAM、viewer、Hist、Raymobtime、AMR-Net_gps_image 或 JEPA-MSAC 路径重新生成为可运行配置
- **AND** retired path MUST 继续 fail fast 或保持普通 missing-file 行为

### Requirement: 配置族必须有 lifecycle inventory
受本 change 影响的 experiment config family MUST 在 inventory、config doctor 或等价文档中声明 lifecycle、owner、是否需要真实数据或本地 checkpoint、默认输出边界、保留理由和删除触发条件。

#### Scenario: family 分类完整
- **WHEN** 开发者查看 Scene31、RBMA missing workflow、strong encoder overlay 或 JEPA image+GPS experiment YAML
- **THEN** 每个保留配置或配置族 MUST 被分类为 canonical/current、paper/workflow reproduction、claim/evidence input、diagnostics manifest、local/manual overlay、generated/recipe-backed、historical 或 delete-candidate
- **AND** local/manual 或 checkpoint-placeholder 配置 MUST 明确不可作为 promoted claim 的充分证据

#### Scenario: config doctor 报告 recipe candidate
- **WHEN** doctor 发现多个实体 YAML 可由同一 generator/manifest 无损重建
- **THEN** doctor MUST 将其报告为 recipe migration candidate 或等价分类
- **AND** 删除必须等 focused tests 和 docs/provenance 更新完成后进行
