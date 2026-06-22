## ADDED Requirements

### Requirement: 高级实体配置删除必须先分类
删除或迁移高级实体 YAML 前，项目 MUST 维护候选分类。每个候选配置 MUST 被归入 canonical/root 保留、recipe 可无损生成、recipe 可生成但有显式差异、人工样例、debug/smoke、diagnostics manifest、历史归档或删除。未分类配置 MUST 不得删除。

#### Scenario: JEPA image GPS 配置矩阵分类
- **WHEN** 开发者准备收敛 `configs/fusion/experiments/jepa_image_gps/*.yaml`
- **THEN** 每个实体 YAML MUST 有分类、保留/删除理由和替代 recipe、overlay、manifest 或文档路径
- **AND** 删除后的 README、docs、tests、scripts 和 OpenSpec current specs MUST 不引用不存在的 current 配置路径

#### Scenario: diagnostics manifest 保留
- **WHEN** 某个 YAML 是手工维护的 diagnostics manifest 且包含 checkpoint 占位、suite 定义或比较矩阵
- **THEN** 本 change MUST 保留该实体 YAML 或提供等价 manifest generator
- **AND** 删除前 MUST 有 focused test 验证 manifest 解析和输出 schema

### Requirement: 可生成配置必须有等价验证
实体 YAML 被 recipe、overlay 或 manifest generator 替代前，项目 MUST 用 focused test 或脚本验证关键解析语义。关键语义 MUST 至少覆盖 experiment name、task/objective、dataset type、enabled modalities、model type、loss type、training defaults、output run name 和 checkpoint/artifact policy。

#### Scenario: 删除可 recipe 化 YAML
- **WHEN** 某个实体 YAML 被分类为 recipe 可无损生成
- **THEN** config load focused test MUST 证明生成配置的关键字段与实体配置等价或列出允许差异
- **AND** 删除后 `final_config.yaml` 和 `resolved_config.yaml` MUST 仍能保存完整解析配置

#### Scenario: 非声明路径仍拒绝
- **WHEN** 用户加载未声明 recipe/overlay 的缺失 YAML
- **THEN** 配置加载 MUST 抛出清晰 `FileNotFoundError` 或 retired-route 错误
- **AND** 系统 MUST 不把任意缺失实验 YAML 自动接管为 virtual config

### Requirement: 重复小工具使用单一 config owner
配置 recipe、CLI overlay 和 model summary 需要 deep merge 时 MUST 使用单一 owner helper。项目 MUST 不保留多个行为近似但 copy 语义不同的 `deep_merge` 实现。

#### Scenario: 删除 recipe deep merge 副本
- **WHEN** canonical recipe 需要 deep merge
- **THEN** 代码 MUST 使用 `kd_sensing.config.io.deep_merge` 或迁移后的单一 owner
- **AND** `kd_sensing.config.canonical_recipes.common.deep_merge` MAY 被删除
