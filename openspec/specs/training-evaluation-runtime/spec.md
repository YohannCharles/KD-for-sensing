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
项目 MUST 为训练编排和配置加载重构提供快速 characterization 检查，覆盖关键输出契约、config load 顺序、十个 retained CLI help 和架构边界。检查 MUST 使用 `kd_mm_beam` 环境，并 MUST 不依赖真实数据、长时间训练或把新 checkpoint 纳入源码。

#### Scenario: 训练短流程 characterization
- **WHEN** 开发者运行训练短流程测试
- **THEN** 测试 MUST 完成 forward、loss、backward、validation、checkpoint 和 artifact 写出
- **AND** 关键输出字段 MUST 保持兼容

#### Scenario: Config load characterization
- **WHEN** 开发者运行 config loading focused tests
- **THEN** 测试 MUST 覆盖实体 YAML、virtual canonical config、migration guard 和命令行覆盖
- **AND** normalization 与 validation MUST 保持兼容

#### Scenario: CLI help characterization
- **WHEN** 开发者运行 CLI help focused tests
- **THEN** 测试 MUST 覆盖 pyproject 声明的十个 retained commands
- **AND** MUST 不要求 project surface doctor、research dashboard 或 research preview

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

### Requirement: early-stopped run 状态识别
summary 和训练产物 MUST 能区分跑满完成、early stopping 正常退出、有 checkpoint 但未完成、以及失败或被 kill 的 run。early stopping 判据 MUST 支持 metrics、训练日志或 run metadata 中的显式标记。

#### Scenario: early stopped 不标为失败
- **WHEN** run 未达到 expected epochs 但存在 `early_stopped=true`、`Early stopping triggered` 或等价 metadata
- **THEN** summary MUST 将状态标记为 `completed_early_stopped`
- **AND** 输出 MUST 包含 best epoch、final epoch、early stop epoch、early stopped 和 expected epochs 字段

#### Scenario: checkpoint 存在但无 early stop 标记
- **WHEN** run 存在 checkpoint 但 final epoch 小于 expected epochs 且没有 early stop 标记
- **THEN** summary MUST 标记为 `incomplete_has_checkpoint`
- **AND** 不得把该状态与 `killed_or_failed` 混淆

### Requirement: hard pattern CE reweight
训练 runtime MUST 支持 sample-wise hard pattern loss weight。启用 `use_pattern_loss_weight=true` 时，系统 MUST 根据 `pattern_loss_weights` 对 CE loss 加权，默认 `apply_pattern_weight_to_ce=true` 且 `apply_pattern_weight_to_proto=false`。

#### Scenario: 只加权 CE
- **WHEN** `radar_only` 配置权重为 1.5 且 `apply_pattern_weight_to_proto=false`
- **THEN** radar_only 样本的 CE loss MUST 乘以 1.5
- **AND** prototype loss MUST 不因该 pattern weight 改变

#### Scenario: metrics 记录 sample weight
- **WHEN** pattern loss weight 启用
- **THEN** metrics MUST 记录 `ce_loss`、`weighted_ce_loss`、`proto_loss` 和 `avg_sample_weight`

### Requirement: mask-conditioned fusion adapter
模型 runtime MUST 支持 opt-in mask-conditioned adapter。启用 `use_mask_adapter=true` 时，adapter MUST 接收 available mask `[B, M]`，通过轻量 MLP 输出与 fused hidden dim 一致的 gamma/beta，并在 fusion 后 beam head 前调制 fused feature。

#### Scenario: 未启用 adapter 保持旧行为
- **WHEN** 配置未声明或设置 `use_mask_adapter=false`
- **THEN** 模型 forward 和 checkpoint shape MUST 与旧配置保持兼容

#### Scenario: adapter 参数量记录
- **WHEN** `use_mask_adapter=true`
- **THEN** startup/runtime metadata MUST 记录 adapter 参数量

### Requirement: weak-pattern KD
训练 runtime MUST 支持 opt-in weak-pattern KD。启用 `use_weak_pattern_kd=true` 时，系统 MUST 对 `kd_apply_patterns` 内样本使用 full modality same-model stopgrad teacher logits，并只对这些样本计算 KD loss。

#### Scenario: KD 只作用于指定 pattern
- **WHEN** `kd_apply_patterns=["radar_only", "lidar_only"]`
- **THEN** 只有 radar_only 和 lidar_only 样本贡献 KD loss
- **AND** eval 时 MUST 不启用 teacher branch

#### Scenario: KD diagnostics
- **WHEN** weak-pattern KD 启用
- **THEN** metrics MUST 记录 `kd_loss` 和 `kd_active_ratio`

### Requirement: lightweight latent prediction probe
训练 runtime MUST 支持 opt-in lightweight latent prediction probe。启用 `use_light_latent_pred=true` 时，系统 MUST 对指定 pattern 的 partial fused feature 预测 stopgrad full fused latent 或 prototype distribution，并只作为 auxiliary loss 使用。

#### Scenario: latent prediction 不插回 fusion
- **WHEN** latent predictor 产生 `h_pred` 或 `q_pred`
- **THEN** 预测结果 MUST 不替换 fused feature 或 beam head 输入
- **AND** eval 时 MUST 不启用 predictor

#### Scenario: latent prediction diagnostics
- **WHEN** latent prediction 启用
- **THEN** metrics MUST 记录 `latent_pred_loss` 和 `latent_pred_active_ratio`

### Requirement: Training runtime is organized into auditable phases
训练 runtime MUST 将 `cfg` 到 run 资源的构建、训练循环、checkpoint、validation、final evaluation、artifact 写出和 shutdown/finalization 拆成可审计 phases。Public `train(cfg)` 行为 MUST 保持兼容，但内部 MUST 使用 run context 或等价结构表达共享状态，避免 `_train_inner` 继续吸收新 workflow 逻辑。

#### Scenario: Run context preserves behavior
- **WHEN** training wave 引入 `TrainingRunContext` 或等价结构
- **THEN** run directory、status file、artifact writer、dataloaders、normalization artifacts、device/model/optimizer/scheduler/scaler、checkpoint manager、TensorBoard writer、extension 和 early stopping state MUST 可从 context 追踪
- **AND** `train_log.json`、`final_config.yaml`、checkpoint layout 和 runtime metadata MUST 保持兼容

#### Scenario: 新训练扩展不修改主循环
- **WHEN** 新增 current training extension、auxiliary loss、metadata handoff 或 final evaluation 行为
- **THEN** 实现 MUST 优先落在 extension、phase helper、runtime metadata helper 或 evaluation owner
- **AND** 不得向 `_train_inner` 添加 suite-specific 大段私有 helper

### Requirement: Evaluation pass is split by schema responsibility
共享 evaluation pass MUST 将 batch iteration、difficulty application、model step、objective label preparation、output recording、metadata recording、metric aggregation 和 prediction artifact schema 拆为职责明确的 helper。拆分 MUST 不改变 validator、evaluator、diagnostics real-forward 和 final-test evaluation 的 public output schema。

#### Scenario: 评估输出 schema 兼容
- **WHEN** `run_evaluation_pass` 内部被拆分
- **THEN** validation metrics、prediction records、objective outputs、metadata rows、difficulty replay metadata 和 diagnostics payload MUST 与变更前兼容
- **AND** `validator.validate`、`evaluator.evaluate` 和 diagnostics real-forward MUST 继续复用同一 shared evaluation pass

#### Scenario: 新 objective 不复制 evaluation loop
- **WHEN** 新增或修改 prediction objective、auxiliary target 或 metric
- **THEN** 实现 MUST 更新 objective metadata、batch labels、loss/metric helper 和 evaluation schema helper
- **AND** 不得新增模型或 objective 专属 validation loop 来绕开 shared evaluation pass

### Requirement: Runtime finalization remains failure-safe
训练和评估 runtime 拆分后 MUST 保持 failure status、dataloader shutdown、TensorBoard close、checkpoint finalization 和 artifact flush 的失败安全语义。异常路径 MUST 继续写出可定位的 failed status，且不得吞掉原始异常。

#### Scenario: 失败路径保持可诊断
- **WHEN** training 或 evaluation phase 抛出异常
- **THEN** runtime MUST 尝试写入 failed status 并关闭可关闭资源
- **AND** 原始异常 MUST 继续向调用方传播，不能被 cleanup/finalization 异常覆盖

### Requirement: Training 与 evaluation runtime 必须保持阶段边界
Training 和 evaluation runtime 重构 MUST 保持 context preparation、resource construction、state restore、epoch loop、evaluation step、metric aggregation 和 finalization 这些显式阶段，并保持公开行为稳定。

#### Scenario: training context 拆分
- **WHEN** training context preparation or resource construction is refactored
- **THEN** run directory creation, initial config artifacts, normalization artifacts, startup summary, AMP, non-blocking transfer and resume validation MUST remain compatible

#### Scenario: evaluation pass 拆分
- **WHEN** evaluation pass internals are split
- **THEN** `EvaluationPassResult`, objective metadata, prediction metadata, metric keys and difficulty stage scoping MUST remain compatible

### Requirement: 单模态 runtime 按同名 modality 路由 input profile
训练、验证和评估共享 runtime MUST 在单模态任务中使用与任务同名的 `model_cfg.input_profiles` 条目准备输入。`radar` MUST 使用 `input_profiles.radar`，`gps` MUST 使用 `input_profiles.gps`，`lidar` MUST 使用 `input_profiles.lidar`，`mmwave` MUST 使用 `input_profiles.mmwave`，`csi` MUST 使用 `input_profiles.csi`。缺省 profile 仍由对应 modality helper 或 modality contract 解析，不得通过读取其它 modality 的 profile 来补偿。

#### Scenario: radar 单模态使用 radar profile
- **WHEN** runtime 准备 `task: radar` 的单模态 batch，且 `model_cfg.input_profiles` 同时包含 `radar`、`gps` 和 `lidar`
- **THEN** `prepare_radar_inputs` MUST 接收 `input_profiles.radar`
- **AND** runtime MUST NOT 读取 `input_profiles.gps` 或 `input_profiles.lidar` 作为 radar profile

#### Scenario: gps 单模态使用 gps profile
- **WHEN** runtime 准备 `task: gps` 的单模态 batch，且 `model_cfg.input_profiles` 同时包含 `gps` 和 `lidar`
- **THEN** `prepare_gps_inputs` MUST 接收 `input_profiles.gps`
- **AND** runtime MUST NOT 读取 `input_profiles.lidar` 作为 gps profile

#### Scenario: 未声明 profile 时保持默认解析
- **WHEN** 单模态 runtime 准备 batch 且 `model_cfg.input_profiles` 缺少对应 modality
- **THEN** runtime MUST 将缺省 profile 交给对应 modality helper 处理
- **AND** 系统 MUST 不因其它 modality profile 存在而改变该任务的默认 profile

### Requirement: Same-model temporal superset consistency
训练 runtime MUST 支持显式 opt-in 的 same-model temporal superset consistency。Training extension MUST 对同一样本使用 `M- subseteq M+` 的 partial student 与 stop-gradient superset teacher，共享一次 teacher forward 给所有启用的 consistency loss；teacher MUST 临时使用 eval mode，结束后恢复 primary model 状态，evaluation MUST 不执行 teacher branch。

#### Scenario: Superset teacher 无外部模型
- **WHEN** temporal superset consistency 启用
- **THEN** runtime MUST 只构建一个 primary model
- **AND** teacher output MUST 来自该 model 的在线 no-grad forward
- **AND** 系统 MUST 不读取 teacher checkpoint 或构建 distiller

#### Scenario: Disabled path 零开销
- **WHEN** KD、beam ranking 和其它 superset loss 均关闭
- **THEN** temporal operator MUST 不保存 superset input payload
- **AND** training extension MUST 不执行第二次 model forward

### Requirement: Confidence-gated soft-logit consistency
系统 MUST 支持温度化 soft-logit KL，并以 stop-gradient teacher correctness 和归一化预测熵形成每样本权重。Teacher 预测错误时权重 MUST 为零；高熵样本权重 MUST 不高于低熵样本；feature L2 MUST 保持独立且 S1 profile 中为零。

#### Scenario: 错误 teacher 不施加强一致性
- **WHEN** superset teacher Top1 与真实标签不同
- **THEN** 该样本对 confidence-gated KL 的权重 MUST 为零
- **AND** diagnostics MUST 记录 gate mean、active ratio、raw KL 和 weighted KL

#### Scenario: Temperature scaling 合法
- **WHEN** `temperature=2` 且至少一个 teacher 样本通过 gate
- **THEN** KL MUST 使用 teacher probability 与 student log-probability
- **AND** loss MUST 乘以 `temperature^2` 并按有效 gate 权重归一

### Requirement: Circular beam-risk monotonic ranking
系统 MUST 支持基于 circular beam distance 的 superset-to-partial ranking。对 64 beam 或配置类别数，风险 MUST 为预测概率对 `min(|b-y|, C-|b-y|)` 的期望；当 superset teacher stop-gradient 时，ranking loss MUST 为 `relu(R(M-) - R(M+) - tolerance)`，不得使用会在激活区间增大 partial student 风险的反向 hinge。

#### Scenario: Circular wraparound 正确
- **WHEN** 真实 beam 为 0 且候选 beam 为 `C-1`
- **THEN** circular distance MUST 为 1
- **AND** 不得使用线性距离 `C-1`

#### Scenario: Monotonic diagnostics
- **WHEN** beam ranking 启用
- **THEN** metrics MUST 记录 ranking loss、teacher/student risk、`student-teacher` risk gap、partial excess violation rate 和只读 superset-worse rate
- **AND** ranking weight 为零时 MUST 不改变 total loss

#### Scenario: 激活 ranking 降低 student 风险
- **WHEN** partial student 风险超过 superset teacher 风险与 tolerance 且执行一步有效梯度更新
- **THEN** 更新后的 student circular risk MUST 下降
- **AND** teacher logits MUST 保持 stop-gradient

### Requirement: Superset 方法保持 extension 边界
temporal superset teacher、confidence gate、beam ranking 和 diagnostics MUST 实现在 U-Mask training extension 或窄 helper 中，不得扩写 trainer/validator 主循环或复制 evaluation loop。

#### Scenario: 共享训练生命周期不增加 suite 分支
- **WHEN** S1 T1/T2/J1 配置运行
- **THEN** trainer MUST 仍通过通用 `TrainingExtension` hooks 调用新增行为
- **AND** checkpoint、optimizer、validation 和 finalization schema MUST 保持兼容

### Requirement: Model selection 与 final test 隔离
训练 runtime MUST 只使用独立 validation split 执行 epoch validation、scheduler、checkpoint 选择和 early stopping。系统 MUST NOT 在 validation 缺失时回退到 test。

#### Scenario: Early stopping 缺少 validation
- **WHEN** resolved config 启用 early stopping 或基于验证指标的 best-checkpoint 选择，但 dataloader 没有独立 validation
- **THEN** 训练 MUST 在第一个 optimizer step 前失败
- **AND** 错误 MUST 提示提供独立 validation 或改用显式 fixed-epoch/no-selection

#### Scenario: Fixed epoch 无 validation
- **WHEN** resolved config 显式使用 fixed epoch、关闭 model selection 且没有 validation
- **THEN** trainer MUST 跳过逐轮 validation、验证 scheduler 和 best-checkpoint 选择
- **AND** final evaluation MUST 使用 `last.pth` 或显式指定 checkpoint
- **AND** test loader MUST 不在训练循环中被迭代

#### Scenario: 独立 validation 正常选模
- **WHEN** dataloader 提供独立 validation 且 resolved config 启用 model selection
- **THEN** trainer MUST 只用 validation metrics 更新 best checkpoint 和 early stopping
- **AND** final test MUST 仅由显式 final evaluation 消费

### Requirement: Validation loss 按有效观测加权
共享 evaluation pass MUST 按每个 batch 的有效 sample 或 token 数聚合 loss，MUST NOT 计算未加权的 batch mean 平均值。

#### Scenario: 最后一个 batch 较小
- **WHEN** validation dataset 的最后一个 batch 小于其它 batch
- **THEN** reported validation loss MUST 等于所有有效观测 loss 总和除以有效观测总数
- **AND** 它 MUST 不因 batch 分组方式不同而改变

#### Scenario: 任务具有有效 token mask
- **WHEN** objective loss 只对部分 token、target 或样本有效
- **THEN** evaluation pass MUST 使用该 objective 的有效计数作为分母
- **AND** 零有效计数 MUST 被清晰拒绝或报告为 unavailable

