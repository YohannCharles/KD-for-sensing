## ADDED Requirements

### Requirement: 高级配置矩阵优先使用 recipe
高级 fusion、objective、G2D、CRAF、MARF 和 ablation 配置矩阵 MUST 优先由 canonical recipe 或 overlay recipe 生成。实体 YAML MUST 只保留无法由 recipe 无损表达、需要人工编辑作为 base/example、或仍处于明确迁移窗口的配置。

#### Scenario: 缺失高级 overlay 配置可生成
- **WHEN** 用户加载已声明支持的高级 fusion overlay 配置路径且磁盘上不存在实体 YAML
- **THEN** 配置加载器 MUST 通过 recipe 生成完整配置
- **AND** 训练、评估和 dry-run 工作流 MUST 像实体 YAML 一样消费该配置

#### Scenario: 保留实体 YAML 优先
- **WHEN** 用户加载磁盘上仍存在的高级 fusion YAML
- **THEN** 配置加载器 MUST 使用实体 YAML 内容
- **AND** 同名 recipe MUST 不覆盖用户在实体 YAML 中显式维护的字段

#### Scenario: 删除实体 YAML 后 final config 完整
- **WHEN** 用户通过 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** `final_config.yaml` 和 `resolved_config.yaml` MUST 保存完整解析配置
- **AND** 运行产物 MUST 不依赖原始 YAML 文件继续存在

### Requirement: 可生成配置删除前必须有等价检查
删除实体配置前，项目 MUST 有 focused test 或脚本验证替代 virtual/overlay 配置的关键语义。关键语义 MUST 至少覆盖 experiment name、task、dataset type、enabled modalities、model type、distillation/loss type、training schedule、output run name 和 checkpoint 来源。

#### Scenario: 关键字段等价
- **WHEN** 开发者准备删除一个可生成实体 YAML
- **THEN** 测试 MUST 比较删除前实体配置和 recipe 生成配置的关键字段
- **AND** 允许差异 MUST 在测试或设计文档中显式列出

#### Scenario: 非 canonical 缺失文件仍被拒绝
- **WHEN** 用户加载未声明 recipe 的缺失 YAML 路径
- **THEN** 系统 MUST 抛出清晰 `FileNotFoundError` 或迁移错误
- **AND** 系统 MUST 不把任意缺失 YAML 自动当作高级 overlay 配置

#### Scenario: 配置矩阵不重新实体化
- **WHEN** 开发者运行配置表面积回归检查
- **THEN** 检查 MUST 拒绝新增与已支持 recipe 等价的实体 YAML
- **AND** 如需新增实体 YAML，必须在 OpenSpec 中说明不能由 recipe 表达的字段
