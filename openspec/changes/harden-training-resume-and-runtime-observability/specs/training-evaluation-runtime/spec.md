## ADDED Requirements

### Requirement: Resume runtime state 必须可重建下一训练步
current checkpoint MUST 保存并恢复版本化 `runtime_state`，其内容 MUST 足以在相同受支持软件/设备拓扑和不可变配置下重建下一 iterator、下一 batch 和下一 optimizer step。状态 MUST 使用 safe checkpoint loader 可读取的 tensor、标量、列表和 mapping 表达，不得要求任意对象 pickle。

#### Scenario: Epoch checkpoint 捕获完整运行时状态
- **WHEN** 训练完成一个 epoch 并发布任一可恢复 checkpoint
- **THEN** `runtime_state` MUST 包含 Python、NumPy、Torch CPU 和所有可见 CUDA device 的 RNG state
- **AND** `runtime_state` MUST 包含按 split 命名的 DataLoader generator、有状态 sampler、GradScaler、每个 active extension、history 和 epoch logs state
- **AND** 无状态 extension MUST 显式记录 stateless，而不是省略后由恢复流程猜测

#### Scenario: 恢复顺序早于下一 iterator
- **WHEN** current checkpoint 通过 schema 与 fingerprint 校验
- **THEN** 系统 MUST 恢复模型、optimizer、scheduler、GradScaler 和 extension state
- **AND** 系统 MUST 在创建下一 train iterator 前恢复全局 RNG、DataLoader generator 和 sampler state
- **AND** 恢复期间用于构建资源的随机消耗 MUST 不改变恢复后的下一 batch 轨迹

#### Scenario: Extension state 不可恢复
- **WHEN** active extension 在 checkpoint 中缺少必需 state、stable id 或不支持其 state schema version
- **THEN** current exact resume MUST 失败并指出 extension id 与缺失字段
- **AND** 系统 MUST 不静默重置该 extension

#### Scenario: 中断恢复与连续训练等价
- **WHEN** deterministic synthetic fixture 在同一受支持环境中分别运行连续 N epoch 和 K epoch 加 current checkpoint resume 到 N epoch
- **THEN** 两条路径的后续 sample/batch 顺序、模型、optimizer、scheduler、GradScaler、extension state MUST 一致
- **AND** history、epoch logs、selection state 和最终选择 checkpoint provenance MUST 一致

### Requirement: Final test 必须加载统一解析的实际选择 checkpoint
训练 runtime MUST 通过单一 selected-checkpoint resolver 决定 final test、run status、返回值和最终 artifact 使用的文件。resolver MUST 使用配置选择策略、逐文件 sidecar、恢复 selection catalog 和 digest，不得按固定文件名猜测。

#### Scenario: 默认 objective selection
- **WHEN** model selection 启用且用户未显式配置独立 checkpoint selection metric
- **THEN** final test MUST 使用 objective/early-stopping 选择的 `best.pth` 或其已验证跨 run 引用
- **AND** selected-checkpoint provenance MUST 记录 metric、mode、value、epoch、role 和 digest

#### Scenario: 显式自定义 selection
- **WHEN** 用户显式配置 `avg_missing_top1`、`worst_pattern_top1`、Top-1 或其它受支持 checkpoint selection
- **THEN** final test MUST 加载与该策略匹配且 sidecar 已验证的 checkpoint
- **AND** 系统 MUST 不固定回退到 `best.pth`

#### Scenario: Fixed epoch 不选模
- **WHEN** model selection 关闭
- **THEN** final test MUST 使用当前 run 的 `last.pth` 或零剩余 epoch 时已验证的 resume checkpoint
- **AND** provenance MUST 将 selection role 标记为 last/fixed-epoch 而不是 best

#### Scenario: 跨 run 且没有新 best
- **WHEN** 训练从另一 run 恢复，且恢复后没有执行 epoch或没有产生新的更优 checkpoint
- **THEN** resolver MUST 使用恢复 selection catalog 中与当前策略匹配的源 checkpoint
- **AND** 它 MUST 验证文件存在且 digest 匹配，并记录 source run
- **AND** 候选缺失、摘要不符或策略歧义 MUST 清晰失败

### Requirement: Final test metrics 必须在 provenance 完整后独立发布
训练结束的 test evaluation MUST 复用共享 evaluation pass，但 MUST 禁止该调用直接覆盖通用/validation `metrics.json`。系统 MUST 在内存中补齐 split、objective 和实际 selected checkpoint provenance 后，原子发布独立 `final_test_metrics.json`，并让 train log、final config 与返回值引用同一内容。

#### Scenario: 最终测试不覆盖 validation metrics
- **WHEN** final test 调用共享 evaluation pass
- **THEN** evaluation pass MUST 返回指标而不把未标注的 test 指标写到 validation 所有的 `metrics.json`
- **AND** 既有 validation metrics artifact MUST 保持原内容

#### Scenario: 写出前补齐测试标签
- **WHEN** final test 指标计算成功
- **THEN** 系统 MUST 在任何最终测试文件写出前补齐 `evaluation_split: test`、model-selection split 和 selected checkpoint path/role/digest/source
- **AND** `final_test_metrics.json`、`train_log.json`、final config 和训练返回值中的对象 MUST 语义一致

#### Scenario: 最终测试失败不标完成
- **WHEN** selected checkpoint 加载、test evaluation 或 final metrics 发布失败
- **THEN** 系统 MUST 不写出声称完整的 final-test artifact
- **AND** run status MUST 不得变为 `complete`
- **AND** 原始异常 MUST 传播给调用方

### Requirement: 训练 epoch 指标按有效观测加权且最小化设备同步
训练 loss、accuracy 和 auxiliary metric MUST 按各自有效 sample 或 token 的 numerator/denominator 聚合，不得按 batch 等权。默认 batch 热路径 MUST 累计 detached device tensors，并仅在进度刷新或 epoch 结束时批量物化紧凑标量，不得为每个指标分别 `.item()`/`.cpu()`。

#### Scenario: 最后一个训练 batch 较小
- **WHEN** train dataset 的最后一个 batch 小于其它 batch
- **THEN** epoch train loss 与 accuracy MUST 等于全部有效观测的总 numerator 除以总 denominator
- **AND** 结果 MUST 不随 batch 分组改变

#### Scenario: 训练目标具有有效 token mask
- **WHEN** 某项训练 loss 只对部分 token、target 或样本有效
- **THEN** 该项指标 MUST 使用自己的有效计数
- **AND** 系统 MUST 不复用 batch size 或另一 loss 的分母
- **AND** 零有效计数 MUST 使用 unavailable 语义或清晰拒绝

#### Scenario: 默认热路径批量搬运标量
- **WHEN** progress、debug 和 timing profile 均未要求当前 batch 刷新
- **THEN** recorder MUST 在设备侧累计 detached numerator/denominator
- **AND** 系统 MUST 不对 total/task/auxiliary loss 和 accuracy 逐项执行同步设备到主机转换

#### Scenario: 进度显示按间隔刷新
- **WHEN** progress 启用并到达配置的刷新间隔
- **THEN** 系统 MUST 一次批量物化用于显示的紧凑标量集合
- **AND** 该显示采样 MUST 不改变最终 epoch 加权指标

### Requirement: 跳过 validation 的 epoch 不得伪造当前观测
当 validation interval 跳过某 epoch 时，当前 epoch 的 validation 字段 MUST 使用既有 null/NaN unavailable 语义。最近一次真实 validation MAY 作为独立 provenance 保存，但 MUST 不写入当前 epoch 的 `val_*` 或用于当前 epoch 的选择决策。

#### Scenario: Validation interval 跳过当前 epoch
- **WHEN** 当前 epoch 未运行 validation 但此前存在 validation 结果
- **THEN** 当前 epoch MUST 记录 `validation_ran: false`
- **AND** 当前 epoch 的 `val_loss`、`val_acc`、`val_primary_metric` 和 `validation_metrics` MUST 为 unavailable
- **AND** 系统 MUST 不把此前结果复制为当前 epoch 结果

#### Scenario: 跳过 validation 不更新选择状态
- **WHEN** 当前 epoch 未运行 validation
- **THEN** checkpoint best selection、scheduler-on-validation、patience 和 early stopping MUST 不更新
- **AND** `last.pth` MAY 记录 `validation_loss: null`，同时 MUST 保留此前真实 validation 产生的 selection catalog

### Requirement: Timing 必须显式启用并声明测量 profile
`training.timing.enabled` MUST 默认解析为 `false`。启用 timing 时配置 MUST 显式声明 host wall-clock 或 CUDA-event profile、采样间隔和测量范围；系统 MUST 将结果缓冲并写入当前 run 专属 artifact，不得默认在共享父路径逐 batch 同步追加。

#### Scenario: 默认关闭 timing
- **WHEN** 配置未声明 timing 或设置 `training.timing.enabled: false`
- **THEN** batch/evaluation 共享路径 MUST 不创建阶段 timer、CUDA event、显存/进程 probe 或 timing 文件
- **AND** 默认训练热路径 MUST 不承担 timing 专用同步开销

#### Scenario: CUDA 阶段 timing
- **WHEN** CUDA 训练显式启用 CUDA-event profile
- **THEN** GPU forward/loss/backward/step 时间 MUST 使用 CUDA event 并只在采样边界同步
- **AND** metadata MUST 记录 profile、采样间隔和观测开销
- **AND** CPU `perf_counter` MUST 不被标注为准确 GPU kernel duration

#### Scenario: Timing 缓冲写入 run 专属 artifact
- **WHEN** timing profile 产生多条 row
- **THEN** 系统 MUST 在内存缓冲并于 epoch/finalization 边界批量写入当前 `run_dir` 下的专属 artifact
- **AND** 多个 run MUST 不同步写同一个父级 timing CSV
- **AND** timing flush 异常 MUST 不覆盖训练原始异常

