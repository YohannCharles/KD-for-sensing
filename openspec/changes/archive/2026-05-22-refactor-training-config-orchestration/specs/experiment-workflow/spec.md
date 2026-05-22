## ADDED Requirements

### Requirement: 训练编排重构保持输出兼容
训练编排内部重构后，训练入口 MUST 保持现有用户可见输出和恢复语义兼容。`final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`training_outputs.npz`、checkpoint、checkpoint sidecar、teacher metrics、TensorBoard events 和 debug artifacts 的关键字段、路径和含义 MUST 与变更前兼容，除非对应 change 明确声明 breaking change。

#### Scenario: 训练日志字段兼容
- **WHEN** 开发者运行 synthetic 或 fixture 短训练并完成至少一个 epoch
- **THEN** `train_log.json` MUST 包含历史兼容的 history 字段、`epoch_logs`、`early_stopping`、`runtime`、`prediction_objective`、`normalization_artifacts` 和 `checkpoint_loads`
- **AND** active objective 的指标字段 MUST 与 objective metadata 声明一致

#### Scenario: training_outputs npz 兼容
- **WHEN** 训练完成并写出 `training_outputs.npz`
- **THEN** 该文件 MUST 包含现有分析脚本依赖的 history 数组、objective 名称、primary loss、primary metric、enabled targets、enabled heads 和 loss weights
- **AND** inactive optional metrics MUST 使用既有 null/NaN 兼容语义表示不可用

#### Scenario: checkpoint metadata 兼容
- **WHEN** 训练保存 `best.pth`、`best_top1.pth` 或 `last.pth`
- **THEN** checkpoint 和 sidecar MUST 继续记录 selection metric、selection mode、selected epoch、objective metric、task metrics、split metadata 和 normalization artifacts
- **AND** 恢复训练 MUST 继续兼容缺少通用 early stopping metadata 的历史 checkpoint

#### Scenario: TensorBoard tag 兼容
- **WHEN** 训练启用 TensorBoard
- **THEN** 系统 MUST 继续写入当前 objective 对应的 TensorBoard scalar tag
- **AND** 用户显式启用 legacy accuracy tags 时，历史 `accuracy/*` 和 `dba/val_adba` tag MUST 继续可选写入

### Requirement: 训练配置重构提供 characterization 检查
项目 MUST 为训练编排和配置加载重构提供快速 characterization 检查，覆盖关键输出契约、config load 顺序、CLI help 和架构边界。检查 MUST 使用 `kd_mm_beam` 环境，并 MUST 不依赖真实数据、长时间训练或新生成 checkpoint 纳入源码。

#### Scenario: 训练短流程 characterization
- **WHEN** 开发者运行本变更记录的训练短流程测试
- **THEN** 测试 MUST 完成 forward、loss、backward、validation、checkpoint 和 artifact 写出
- **AND** 测试 MUST 验证重构后的关键输出字段与兼容契约一致

#### Scenario: config load characterization
- **WHEN** 开发者运行 config loading focused tests
- **THEN** 测试 MUST 覆盖实体 YAML、virtual canonical 配置、snapshot 配置、Raymobtime 配置和命令行覆盖
- **AND** 测试 MUST 验证 normalization 与 validation 结果保持兼容

#### Scenario: CLI help characterization
- **WHEN** 开发者运行 CLI help focused tests
- **THEN** `kd-sensing-train --help`、`kd-sensing-evaluate --help`、`kd-sensing-preprocess --help`、`kd-sensing-export-viewer-manifest --help` 和 `kd-sensing-visualize-modalities --help` MUST 正常退出
- **AND** 检查 MUST 不读取真实数据集、不加载 checkpoint、不启动训练
