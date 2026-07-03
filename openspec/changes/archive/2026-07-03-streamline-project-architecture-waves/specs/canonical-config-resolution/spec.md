## ADDED Requirements

### Requirement: Config surface distinguishes canonical, recipe, generated, and local/manual
配置生命周期 MUST 区分 canonical/current entity YAML、virtual config、recipe/generated config、experiment reproduction/local manual overlay、diagnostics manifest 和 retired config。可生成或本地队列型配置 SHOULD 通过 recipe/manifest/generator 表达；tracked entity YAML MUST 有 current 入口、复现实验、diagnostics manifest 或 local/manual 登记理由。

#### Scenario: 生成型配置不无限实体化
- **WHEN** 新增 Scene31 seed sweep、night-grid、next-round、ablation matrix 或其它规则化配置族
- **THEN** 项目 MUST 优先提供 recipe、manifest 或 generator sanity test
- **AND** 若提交实体 YAML，inventory 或 tasks MUST 说明为何不能只由 recipe 生成，以及该 YAML 的 lifecycle

#### Scenario: Canonical 配置保留实体入口
- **WHEN** 配置属于 README/current docs 推荐的 canonical single-modality、fusion、diagnostic 或 paper/workflow reproduction 入口
- **THEN** 实体 YAML MAY 保留
- **AND** virtual config 或 recipe MUST 不接管 retired KD、BGAM、viewer、Hist、Raymobtime、AMR mock 或 JEPA-MSAC 路径

### Requirement: Generated config recipes preserve resolved semantics
Recipe/generated config MUST 生成与等价实体 YAML 相同的 resolved config 语义，并在 sanity tests 中覆盖 run name、seed、epoch、sampler、loss weights、missing pattern、difficulty profile 和 output boundary 等关键字段。

#### Scenario: Recipe sanity validation
- **WHEN** generator 创建本地实验矩阵
- **THEN** focused tests MUST 校验 manifest 行、文件名/run name 和 resolved config 关键字段一致
- **AND** generator MUST 不写入 `outputs/`、`logs/`、checkpoint 或真实训练结果

### Requirement: Config cleanup keeps migration guards
配置表面瘦身 MAY 删除重复实体 YAML、旧 alias、未登记 local queue config 或退役路径，但 MUST 保留仍有当前迁移价值的 guard、错误信息或 retired summary。

#### Scenario: 删除重复配置
- **WHEN** 一个实体 YAML 可由 current recipe/virtual config 无损生成，且不属于 canonical/current 推荐入口
- **THEN** 本 change MAY 删除该 YAML 或迁到 local/manual generated surface
- **AND** 配置加载器 MUST 继续拒绝 retired config path，不能把旧路径静默映射到新 recipe

