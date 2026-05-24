## ADDED Requirements

### Requirement: 高级配置二次瘦身必须有候选分类
项目 MUST 在删除仍保留的高级实体 YAML 前维护候选分类。每个候选配置 MUST 被归入可由 recipe 无损生成、可由 recipe 生成但存在显式差异、或需要作为人工样例继续保留三类之一。

#### Scenario: 生成配置瘦身候选清单
- **WHEN** 开发者准备收敛 `configs/fusion/`、`configs/csi/hardening_matrix/` 或其它高级实验配置矩阵
- **THEN** 清单 MUST 记录每个候选实体 YAML 的分类、保留或删除理由和对应 recipe/overlay 名称
- **AND** 未分类的实体 YAML MUST 不得被删除

#### Scenario: 有差异的实体配置先记录差异
- **WHEN** 某个实体 YAML 与候选 recipe 在模型、loss、training schedule、dataset 字段或 checkpoint 来源上存在差异
- **THEN** 该差异 MUST 先记录为允许差异、overlay option 或保留理由
- **AND** 不得把该实体 YAML 当作无损可生成配置直接删除

### Requirement: 高级实体 YAML 删除前必须通过等价检查
删除高级实体 YAML 前，项目 MUST 提供 focused test 或脚本比较实体配置和替代 virtual/overlay 配置的关键语义。关键语义 MUST 至少覆盖 experiment name、task、dataset type、enabled modalities、model type、loss/distillation type、training schedule、output run name 和 checkpoint 来源。

#### Scenario: 可生成高级配置关键字段等价
- **WHEN** 开发者删除一个由 recipe 覆盖的 CRAF、MARF、G2D、token transformer、CSI/GPS/mmWave 组合或 ablation 实体 YAML
- **THEN** 等价检查 MUST 证明替代 virtual/overlay 配置的关键字段与原实体 YAML 一致
- **AND** 允许差异 MUST 在测试断言或设计文档中显式列出

#### Scenario: 删除后运行产物保存完整配置
- **WHEN** 用户使用已删除实体 YAML 对应的 virtual/overlay 路径启动 dry-run、训练或评估
- **THEN** 配置加载器 MUST 生成完整最终配置
- **AND** 运行目录中的 `final_config.yaml` 和 `resolved_config.yaml` MUST 不依赖原实体 YAML 继续存在

### Requirement: 高级 overlay recipe 必须按领域拆分
高级配置生成 MUST 将 G2D、CRAF、MARF、objective、CSI hardening 和组合实验 overlay 的主要字段定义放入可审查的 recipe/table 或领域 helper 中。`build_virtual_config()` 和路径识别入口 MUST 只负责识别路径、查找 recipe 和应用 overlay。

#### Scenario: 新增 CRAF 或 MARF ablation 不扩写路径入口
- **WHEN** 开发者新增一个 CRAF 或 MARF ablation 配置 overlay
- **THEN** 主要变更 MUST 位于对应 recipe/table 或领域 helper
- **AND** 不得在 `build_virtual_config()` 中新增大段方法专属字段表

#### Scenario: 新增 CSI 组合配置有明确 recipe 来源
- **WHEN** 开发者新增 CSI/GPS/mmWave 或 CSI hardening 组合配置的 virtual/overlay 支持
- **THEN** recipe MUST 明确声明模态集合、dataset 字段、模型类型、loss/distillation、training 和 output run name
- **AND** 非声明路径 MUST 继续抛出清晰缺失配置错误

### Requirement: 可生成配置不得重新实体化
当某类配置已经由 canonical recipe 或 advanced overlay 无损生成后，项目 MUST 防止等价实体 YAML 重新进入源码表面积。确需新增实体 YAML 时，OpenSpec change MUST 说明 recipe 无法表达的字段或人工样例用途。

#### Scenario: 表面积检查拒绝已支持 recipe 的实体 YAML
- **WHEN** 开发者新增与已支持 virtual/overlay recipe 等价的实体 YAML
- **THEN** 表面积回归检查 MUST 拒绝该文件
- **AND** 错误或测试说明 MUST 指向对应 recipe 路径或要求补充 OpenSpec 保留理由

#### Scenario: 人工样例配置保留原因可审计
- **WHEN** 项目保留一个不能删除的高级实体 YAML
- **THEN** inventory MUST 记录它作为 base、example、迁移窗口或不可 recipe 化实验的用途
- **AND** 后续删除或 recipe 化该文件 MUST 更新对应记录
