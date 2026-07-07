## ADDED Requirements

### Requirement: 实验配置删除必须保留证据链
项目 MAY 删除重复、生成型或历史 local/manual experiment YAML，但 MUST 保留当前 claim/evidence、paper reproduction、diagnostics manifest 和必要 focused tests 的配置证据链。删除说明 MUST 记录替代 generator/manifest/base config、删除原因、引用同步和回滚方式。

#### Scenario: claim 配置被保护
- **WHEN** YAML 被 `docs/result_claims_registry.md`、`docs/mainline_model_catalog.md`、`docs/experiment_matrix.md`、当前 OpenSpec specs 或 focused tests 引用为证据输入
- **THEN** implementation MUST 保留该 YAML，或先更新 claim provenance 指向等价 generator/manifest 输入
- **AND** 删除后 current docs MUST 不指向不存在路径

#### Scenario: historical sweep 配置可删除
- **WHEN** YAML 只服务已沉淀的历史 sweep、local queue 或被 generator 覆盖的 seed/missing-pattern 组合
- **THEN** implementation MAY 删除实体 YAML
- **AND** 有价值的结论、caveat 或复跑方式 MUST 保留在 docs、inventory 或 result provenance 中

#### Scenario: 清理不触碰 runtime artifact
- **WHEN** implementation 收缩 experiment config family
- **THEN** implementation MUST 不删除、移动、重写或纳入 `outputs/`、`logs/`、checkpoint、cache、TensorBoard event 或真实 `dataset/`
- **AND** generator focused tests MUST 使用临时目录或受控源码配置路径，不写入真实训练产物

### Requirement: 配置表面收缩必须同步引用
当实体 YAML 被删除、迁移、生成化或降级为 local/manual 后，README、docs、OpenSpec current specs、scripts 默认路径、tests 和 inventory MUST 同步更新。健康检查 MUST 能发现 current 引用指向不存在配置。

#### Scenario: stale config reference 被发现
- **WHEN** current docs、scripts、tests 或 OpenSpec specs 仍引用已删除 YAML
- **THEN** architecture/config/surface 检查 MUST 失败或报告 error
- **AND** 修复路径 MUST 是恢复配置、更新引用到 generator/manifest，或将引用标记为 historical

#### Scenario: root/canonical surface 不被实验 YAML 污染
- **WHEN** experiment family shrink 后仍保留 local/manual YAML
- **THEN** 该 YAML MUST 位于语义明确的 experiment/local 目录或被 inventory 分类
- **AND** 它 MUST 不被迁入 root canonical config surface，除非 OpenSpec 明确将其提升为 current canonical entry
