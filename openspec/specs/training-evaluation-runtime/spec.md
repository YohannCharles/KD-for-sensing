# training-evaluation-runtime Specification

## Purpose
定义训练、验证、评估 runtime 的 AMP、early stopping、共享 evaluation pass、baseline 回归和配置 characterization 行为，使 experiment workflow 保持入口级契约。

## Requirements

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

### Requirement: 默认 early stopping 指标使用 DBA
训练工作流 MUST 在 `experiment.objective: beam` 或未显式设置 objective 的历史 beam 训练中默认使用验证 DBA/ADBA 作为 early stopping 监控指标。objective-aware 非 beam 训练 MUST 使用对应预测目标的默认主指标：`occlusion` 使用 `val_occlusion_blocked_f1/max`，`position` 使用 `val_position_rmse/min`，`multitask` 使用 `val_multitask_loss/min` 或用户显式配置的可用 multitask 主指标。默认配置 MUST NOT 使用 `top1_val_acc`、`val_acc` 或其它 Top-1 验证准确率别名作为默认 early stopping 指标。

#### Scenario: 默认配置记录 DBA early stopping
- **WHEN** 用户使用未设置 `experiment.objective` 的默认 image、radar、GPS、LiDAR、mmWave 或 fusion 训练配置启动训练
- **THEN** 系统 MUST 将 objective 解析为 `beam`
- **AND** 系统 MUST 在解析后的最终配置中记录 early stopping 监控指标为 `val_adba` 或等价 DBA 别名
- **AND** 系统 MUST 将 early stopping 比较方向记录为越大越好
- **AND** 系统 MUST 不把 `top1_val_acc` 或等价 Top-1 验证准确率别名作为默认 early stopping 指标

#### Scenario: canonical beam 配置默认使用 DBA
- **WHEN** 开发者生成或读取 beam objective canonical 训练配置
- **THEN** canonical 配置 MUST 默认包含 DBA/ADBA early stopping 指标
- **AND** canonical 配置 MUST 不把 Top-1 验证准确率作为默认 early stopping 指标

#### Scenario: objective-aware occlusion 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: occlusion` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_occlusion_blocked_f1`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: max`

#### Scenario: objective-aware position 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: position` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_position_rmse`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`

#### Scenario: objective-aware multitask 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: multitask` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_multitask_loss`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`
- **AND** runtime metadata MUST 记录该 multitask loss 使用的分任务权重

#### Scenario: 显式覆盖 early stopping 指标
- **WHEN** 用户在训练配置或命令行覆盖中显式设置 early stopping 指标为 Top-1、loss 或其它受支持指标
- **THEN** 系统 MUST 使用用户显式指定的指标和比较方向
- **AND** 系统 MUST 校验该指标在当前 objective 的验证结果中真实可用
- **AND** 该覆盖 MUST 不改变项目默认配置继续使用 objective-specific 默认指标的要求

### Requirement: 训练循环按配置指标执行 early stopping
训练循环 MUST 从每个 epoch 的验证标量中解析配置的 early stopping 指标，并基于该指标更新最佳值、patience 计数和默认最佳 checkpoint。DBA/ADBA 和准确率类指标 MUST 按越大越好判断 improvement；loss 类指标 MUST 按越小越好判断 improvement。

#### Scenario: DBA improvement 重置 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 相比历史最佳值提升超过 `training.min_delta`
- **THEN** 系统 MUST 更新最佳 early stopping 值和最佳 epoch
- **AND** 系统 MUST 将 `epochs_without_improvement` 重置为 0
- **AND** 系统 MUST 保存默认最佳 checkpoint

#### Scenario: DBA 未提升累计 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 未提升超过 `training.min_delta`
- **THEN** 系统 MUST 累加 `epochs_without_improvement`
- **AND** 当 `training.use_early_stopping` 启用且累计值达到 `training.patience` 时，系统 MUST 停止训练

#### Scenario: 缺失 DBA 指标时报错
- **WHEN** 默认 early stopping 指标为 DBA/ADBA 但验证结果没有产出可解析的 DBA/ADBA 标量
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 early stopping 指标，并提示用户补齐 DBA 指标或显式配置其它受支持指标

### Requirement: early stopping metadata 可复现
训练产物 MUST 记录实际使用的 early stopping 指标、比较方向、最佳值、最佳 epoch 和未提升 epoch 计数。恢复训练 MUST 优先使用这些通用 metadata 继续 early stopping 状态；历史 checkpoint 缺少通用 metadata 时，系统 MUST 使用兼容路径恢复已有 loss 或 Top-1 相关状态。

#### Scenario: checkpoint 记录 early stopping 状态
- **WHEN** 训练完成至少一个 epoch 并保存 `last.pth`
- **THEN** checkpoint metadata MUST 包含实际 early stopping 指标、比较方向、最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 运行日志或最终配置 MUST 能追溯本次训练使用的 early stopping 指标

#### Scenario: 恢复 DBA early stopping 状态
- **WHEN** 用户从包含通用 early stopping metadata 的 checkpoint 恢复训练
- **THEN** 系统 MUST 恢复 DBA/ADBA 的最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 后续 early stopping 判断 MUST 延续恢复前的指标和比较方向

#### Scenario: 兼容历史 checkpoint
- **WHEN** 用户从缺少通用 early stopping metadata 的历史 checkpoint 恢复训练
- **THEN** 系统 MUST 尽可能从历史 `best_val_loss`、`best_val_top1` 或等价字段恢复状态
- **AND** 系统 MUST 不因缺少新 metadata 而拒绝恢复历史 checkpoint

### Requirement: 单模态 baseline 回归检查
项目 MUST 提供面向 image 和 LiDAR 默认 baseline 的回归检查，防止默认配置重新退回到从头训练 camera encoder 或 LiDAR 多数类退化路径。

#### Scenario: image 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 image teacher/no-KD 配置使用 `resnet18_imagenet_rgb`
- **AND** 测试 MUST 验证该 encoder 配置启用 ImageNet 预训练权重

#### Scenario: LiDAR 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 LiDAR teacher/no-KD 配置显式启用 LiDAR streaming stats normalization
- **AND** 测试 MUST 验证该配置记录可追踪的 BEV ROI/cache 参数

#### Scenario: LiDAR 退化报告回归检查
- **WHEN** 开发者运行 LiDAR 评估或诊断测试
- **THEN** 输出报告 MUST 包含 majority-class baseline
- **AND** 输出报告 MUST 包含 LiDAR input quality summary
- **AND** 报告 MUST 能标记模型未超过 majority-class baseline 的退化风险

### Requirement: 共享 evaluation pass
训练验证、force-mask subset 验证和 standalone evaluate MUST 复用同一个 evaluation pass 完成 batch 准备、model forward、objective loss、输出收集、指标聚合和 available metrics 生成。各入口 MAY 对结果做输出包装或文件写出，但 MUST 不复制核心 forward/loss/collect 逻辑。

#### Scenario: 普通验证使用共享 pass
- **WHEN** 训练流程在 epoch 结束后调用 validation
- **THEN** validation MUST 通过共享 evaluation pass 计算 loss、Top-K、DBA 和 objective 指标
- **AND** 返回的公开 metrics 键 MUST 保持与变更前兼容

#### Scenario: force-mask subset 使用共享 pass
- **WHEN** evaluation 配置启用 modality subset 或 force mask 验证
- **THEN** subset validation MUST 使用同一个 evaluation pass 并传入 mask 选项
- **AND** subset 结果 MUST 包含与普通验证一致的 objective metadata 和 available metrics

#### Scenario: standalone evaluate 使用共享 pass
- **WHEN** 用户通过评估入口运行 checkpoint evaluate
- **THEN** evaluate MUST 使用共享 evaluation pass 计算指标
- **AND** 保存的报告 MUST 与训练验证使用同一套 objective 指标语义

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
- **THEN** 测试 MUST 覆盖实体 YAML、virtual canonical 配置、snapshot 配置、Raymobtime migration guard 和命令行覆盖
- **AND** 测试 MUST 验证 normalization 与 validation 结果保持兼容

#### Scenario: CLI help characterization
- **WHEN** 开发者运行 CLI help focused tests
- **THEN** `kd-sensing-train --help`、`kd-sensing-evaluate --help`、`kd-sensing-preprocess --help`、`kd-sensing-jepa-visual-analysis --help` 和 `kd-sensing-jepa-gps-shortcut-benchmark --help` MUST 正常退出
- **AND** 检查 MUST 不读取真实数据集、不加载 checkpoint、不启动训练

### Requirement: Beam TensorBoard 指标命名空间
训练流程 MUST 为 beam 预测写入 objective-specific TensorBoard 标量命名空间。`beam/*` 标量 MUST 只表示 active beam objective 或 multitask 中的 active beam 分任务，不得包含 occlusion-only 或 position-only 训练中的诊断性 beam accuracy。默认 TensorBoard 输出 MUST 不再依赖通用 `accuracy/*` 分组作为 beam 指标入口；历史通用 tag 只能作为显式兼容路径写入。

#### Scenario: beam objective 写入 beam 指标
- **WHEN** 用户运行 `experiment.objective: beam` 或未显式设置 objective 的历史 beam 训练，并启用 TensorBoard
- **THEN** 训练流程 MUST 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`
- **AND** 这些 tag MUST 分别对应当前 epoch 的 `train_acc`、`val_acc`、`val_atop3`、`val_atop5` 和 `val_adba`
- **AND** 写入前 MUST 跳过缺失、`null`、`NaN` 或非 finite 的值

#### Scenario: occlusion 单任务不污染 beam 指标
- **WHEN** 用户运行 `experiment.objective: occlusion` 的单任务训练，并启用 TensorBoard
- **THEN** 训练流程 MUST NOT 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 或 `beam/val_adba`
- **AND** 即使 validator 能计算诊断性 beam `val_acc`，该值也 MUST NOT 出现在 `beam/*` TensorBoard 命名空间中

#### Scenario: position 单任务不污染 beam 指标
- **WHEN** 用户运行 `experiment.objective: position` 的单任务训练，并启用 TensorBoard
- **THEN** 训练流程 MUST NOT 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 或 `beam/val_adba`
- **AND** position TensorBoard 指标 MUST 继续通过 `position/rmse` 和 `position/mae` 表示

#### Scenario: multitask 写入 active beam 分任务指标
- **WHEN** 用户运行 `experiment.objective: multitask` 且 beam 分任务参与 loss 或主验证指标计算，并启用 TensorBoard
- **THEN** 训练流程 MUST 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`
- **AND** 训练流程 MUST 继续写入 active 的 `occlusion/*` 和 `position/*` 指标

#### Scenario: 默认不写历史通用 accuracy tag
- **WHEN** 用户启用 TensorBoard 且未显式设置 `output.tensorboard.legacy_accuracy_tags: true`
- **THEN** 训练流程 MUST NOT 写入 `accuracy/train`、`accuracy/val`、`accuracy/val_atop3`、`accuracy/val_atop5` 或 `dba/val_adba` 作为默认 beam 指标
- **AND** `train_log.json`、`training_outputs.npz` 和 checkpoint metadata MUST 继续保留既有内部 metric key，便于旧分析脚本读取

#### Scenario: 显式启用历史通用 tag
- **WHEN** 用户设置 `output.tensorboard.legacy_accuracy_tags: true` 并启用 TensorBoard
- **THEN** 训练流程 MAY 额外写入历史 `accuracy/*` 和 `dba/val_adba` tag
- **AND** 这些 legacy tag MUST 被文档标记为兼容入口，不得作为 objective-aware 实验比较的推荐入口

### Requirement: Beam metric alias 兼容
训练流程 MUST 支持 objective-specific beam metric 名称作为 early stopping 和用户配置别名。新增 `beam/*` 别名 MUST 解析到既有内部 metric key，同时历史 `accuracy/*` 和 `dba/*` 别名 MUST 保持可用。

#### Scenario: 使用 beam ADBA tag 配置 early stopping
- **WHEN** 用户将 early stopping metric 配置为 `beam/val_adba`
- **THEN** 系统 MUST 将该配置解析为内部 `val_adba`
- **AND** 比较方向 MUST 支持按 DBA/ADBA 语义使用越大越好

#### Scenario: 使用 beam Top-1 tag 配置 early stopping
- **WHEN** 用户将 early stopping metric 配置为 `beam/accuracy_val` 或 `beam/val_top1`
- **THEN** 系统 MUST 将该配置解析为内部 `val_acc`
- **AND** 比较方向 MUST 支持按 accuracy 语义使用越大越好

#### Scenario: 历史 early stopping 别名继续可用
- **WHEN** 用户将 early stopping metric 配置为 `accuracy/val`、`accuracy/val_top1` 或 `dba/val_adba`
- **THEN** 系统 MUST 继续解析到对应内部 beam metric
- **AND** 解析行为 MUST 不要求 TensorBoard 继续写入同名 legacy tag

### Requirement: 评估指标写出与 runtime metadata 对齐
训练和评估写出的 metrics/report MUST 包含 objective runtime metadata、primary metric、available metrics 和已启用模态信息。该 metadata MUST 来自 objective 与 modality resolution 层，而不是入口各自手写推导。

#### Scenario: 评估报告记录 objective metadata
- **WHEN** 用户评估 `experiment.objective: occlusion` 的模型
- **THEN** 评估报告 MUST 记录 objective 名称、primary loss、primary metric、metric mode、enabled targets 和 enabled heads
- **AND** 这些字段 MUST 与训练 final config 中的 prediction objective metadata 一致

#### Scenario: 评估报告记录启用模态
- **WHEN** 用户评估 GPS+mmWave fusion 模型
- **THEN** 评估报告 MUST 记录启用模态为 `["gps", "mmwave"]`
- **AND** 该模态集合 MUST 由统一模态解析逻辑产生
