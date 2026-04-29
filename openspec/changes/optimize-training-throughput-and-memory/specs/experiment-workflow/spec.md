## ADDED Requirements

### Requirement: 吞吐优化配置与日志
训练配置 MUST 暴露吞吐相关开关，包括 DataLoader worker/prefetch 参数、non-blocking transfer、AMP 和预处理 cache 读取/写入策略。训练日志、最终配置或 profile 输出 MUST 记录这些实际生效的吞吐参数，便于比较不同实验设置。

#### Scenario: 记录吞吐参数
- **WHEN** 一次训练或 profile 运行启动
- **THEN** 输出配置或日志 MUST 记录 `num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、non-blocking transfer、AMP enabled/dtype 和启用的 cache 目录
- **AND** 对启用 image 或 LiDAR 的配置 MUST 记录对应 cache 参数 hash 目录

#### Scenario: 并行实验默认不过度放大 worker
- **WHEN** 用户使用 canonical 单模态或 fusion YAML 运行实验
- **THEN** 配置 SHOULD 使用适合并行实验的保守 `num_workers` 和 `prefetch_factor`
- **AND** 用户 MUST 能通过命令行覆盖这些参数以寻找单实验最高吞吐

### Requirement: AMP 训练配置兼容
训练工作流 MUST 支持通过配置启用或关闭 AMP。AMP 配置 MUST 不影响 checkpoint 保存、早停、scheduler、TensorBoard、registry 和评估指标输出结构。

#### Scenario: 开启 AMP 完成短训练
- **WHEN** 用户在 CUDA device 上启用 AMP 并运行 1 epoch smoke test
- **THEN** 训练 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存
- **AND** 训练日志 MUST 记录 AMP 已启用和实际 dtype

#### Scenario: 关闭 AMP 保持旧行为
- **WHEN** 用户关闭 AMP 或在 CPU device 上运行训练
- **THEN** 训练 MUST 保持现有 FP32 行为
- **AND** 旧配置未声明 AMP 字段时 MUST 能继续运行
