## MODIFIED Requirements

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称或退役研究线组件名称时，注册表错误 MUST 至少包含请求名称、registry 名称或可用名称上下文。对于仍有当前迁移价值的名称，错误 SHOULD 区分“未知名称”和“已删除名称”并给出迁移方向；对于完全退役且不再承诺兼容的历史名称，系统 MAY 使用普通 unknown-name 错误或集中退役说明替代长期 removed guard table。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset 且项目仍保留该迁移说明
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求旧 fusion 类名 alias 或已删除 image encoder alias，且该名称仍在 current migration table 中
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名

#### Scenario: 已退役研究线组件
- **WHEN** 用户请求 `craf_fusion`、`marf_fusion`、`g2d` distiller 或 `multimodal_nf` dataset
- **THEN** 系统 MUST 拒绝构建
- **AND** 系统 MAY 报告为已退役名称或普通未知名称，但 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

### Requirement: 已删除组件错误包含 Hist 迁移方向
当用户引用 Hist 旧组件名且该名称仍由当前迁移 guard 覆盖时，错误信息 MUST 区分退役研究线与普通拼写错误，并指向当前推荐 workflow 或说明无兼容迁移。若本 change 删除对应 guard table，Hist 旧组件名 MAY 回落为普通未知名称，但仍 MUST 不注册为 current 可构建组件。

#### Scenario: Hist 旧模型名错误可诊断
- **WHEN** 用户配置 `model.primary.type: hist_beam_fusion`
- **THEN** 系统 MUST 拒绝构建并包含请求名称
- **AND** 若 Hist guard 被保留，错误信息 MUST 提示使用当前 supervised、adapter、GPS candidate、residual fusion 或其它保留 workflow；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误

### Requirement: JEPA downstream pooler 和 adapter 注册
项目 MUST 通过轻量组件构建边界暴露 JEPA downstream pooler。内置 mean pooler 和 GPS-query attention pooler MUST 能通过配置名称构建；identity adapter MAY 作为默认 no-op 路径内联，而不是必须注册为独立 adapter。未知 pooler 名称 MUST 使用现有 registry 错误风格报告；未知 adapter 名称只有在非 identity adapter 配置面被保留时才需要注册表式错误。

#### Scenario: 按名称构建 mean pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `mean`
- **THEN** 系统 MUST 构建 mean pooler
- **AND** 该 pooler MUST 接收 patch tokens `[B,T,N,D]` 并输出 `[B,T,D]`

#### Scenario: 按名称构建 GPS-query pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `gps_query_attention`
- **THEN** 系统 MUST 构建 GPS-query attention pooler
- **AND** 构建参数 MUST 支持 `k_queries`、`num_heads`、`condition_dim`、`latent_dim`、dropout 和 condition source

#### Scenario: identity adapter 内联为 no-op
- **WHEN** `jepa_context_image` 配置未声明 adapter 或声明 adapter 为 `identity`
- **THEN** 系统 MUST 使用不改变输入 shape 的无操作路径
- **AND** 现有配置 MUST 无需新增 adapter 字段即可运行

#### Scenario: 未知 JEPA downstream 组件可诊断
- **WHEN** 用户配置不存在的 JEPA downstream pooler 名称
- **THEN** 系统 MUST 拒绝构建
- **AND** 错误信息 MUST 包含请求名称、组件类别和可用 pooler 名称

### Requirement: JEPA downstream 注册保持轻量导入
JEPA downstream pooler 的注册 MUST 不破坏 registry 轻量导入边界。导入 `kd_sensing.registries` MUST 不 eager import torch model implementation、dataset、diagnostics、训练器或 checkpoint 文件；默认组件导入流程 MUST 显式注册内置 JEPA downstream pooler。identity adapter 若内联为 no-op，则不需要默认注册流程。

#### Scenario: 轻量导入 registry 不触发 JEPA model
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import `kd_sensing.models.jepa` 或 JEPA downstream 实现模块

#### Scenario: 默认组件导入后可构建 JEPA downstream pooler
- **WHEN** 构建流程调用默认组件导入函数
- **THEN** 内置 JEPA downstream pooler MUST 完成注册
- **AND** 用户配置中的内置 pooler 名称 MUST 可解析

### Requirement: Legacy model registry names are retired with migration guards
项目 MUST 将已退役的 legacy model、encoder、core 和 head 注册名排除在 current 可构建组件之外。对仍有当前迁移价值的旧名称，项目 MAY 保留 removed guard 并给出明确迁移目标；对完全退役且不再承诺兼容的旧名称，项目 MAY 删除 guard table 并让 registry 使用普通 unknown-name 错误。

#### Scenario: 旧整模型注册名被拒绝
- **WHEN** 用户通过 `MODELS.build()` 请求 `radar_strong`、`gps_lightweight`、`mmwave_strong`、`fusion_lightweight` 或其它本 change 退役的旧整模型注册名
- **THEN** 系统 MUST 拒绝构建该名称
- **AND** 若 removed guard 被保留，错误信息 MUST 包含请求名称、registry 名称和 `modular_sequence` 迁移目标；若 guard 已删除，系统 MAY 使用普通 unknown-name 错误

#### Scenario: 旧别名被拒绝
- **WHEN** 用户请求 `modular_sequence_model`、`gps_only_neural_baseline`、`jepa_token_transformer` 或 `safe_residual_reranker`
- **THEN** 系统 MUST 不把这些名称注册为 current 可构建组件
- **AND** 若别名仍在 current migration table 中，错误信息 MUST 指向对应 canonical 名称或配置路径

#### Scenario: feature extractor 不作为完整模型列出
- **WHEN** 默认组件导入完成后开发者查看 `MODELS.list()`
- **THEN** 输出 MUST NOT 包含 `radar_feature_extractor`、`lidar_feature_extractor` 或 `mmwave_feature_extractor`
- **AND** 对应 feature extractor 类 MAY 继续通过窄模块导入或由 encoder 组件内部复用

#### Scenario: current registry discovery 只列当前入口
- **WHEN** 文档、架构摘要或架构边界测试检查 current registry surface
- **THEN** current model/encoder/core/head 清单 MUST 不把 removed guard 名称展示为可推荐入口
- **AND** removed 名称 MAY 出现在退役边界、migration table 或普通错误路径中
