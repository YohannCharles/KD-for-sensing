## MODIFIED Requirements

### Requirement: Teacher metrics export
单模态 teacher 训练产物 MUST 提供 teacher registry 可读取的指标文件或等价 metadata。指标 MUST 至少包含模态、selected/best epoch、验证 Top-1、验证 Top-3、验证 Top-5、验证 ADBA、训练 Top-1、checkpoint 路径、checkpoint 来源、selection metric 和 selection mode。`best_epoch` MUST 指向 registry 默认应使用的 selected checkpoint epoch；如果系统同时保存最高 Top-1 checkpoint，指标 MUST 另行记录 Top-1 checkpoint 或 Top-1 epoch，不得用 Top-1 epoch 覆盖 objective-selected epoch。

#### Scenario: teacher 训练完成写出 metrics
- **WHEN** 单模态 teacher 训练完成至少一个 epoch
- **THEN** 输出目录 MUST 包含可供 registry 构建脚本读取的指标数据
- **AND** 指标数据 MUST 包含 `modality`、`best_epoch`、`val_acc_top1`、`val_acc_top3`、`val_acc_top5`、`val_adba` 和 `train_acc_top1`
- **AND** 指标数据 MUST 包含 `selection_metric`、`selection_mode`、`checkpoint` 或等价 checkpoint 路径字段
- **AND** `best_epoch` MUST 与 `checkpoint` 对应的 epoch 一致

#### Scenario: early stopping objective 作为默认 teacher 指标选择
- **WHEN** 训练流程保存了 early stopping objective 对应的 `best.pth` 且用户未显式要求 Top-1 teacher
- **THEN** teacher metrics MUST 使用 early stopping objective 对应的 epoch 作为 `best_epoch`
- **AND** teacher metrics MUST 将 checkpoint 路径记录为 `best.pth` 或其归档副本
- **AND** teacher metrics MUST 仍可记录最高 Top-1 epoch 作为附加诊断字段

#### Scenario: 显式 Top-1 teacher 指标选择
- **WHEN** 用户或配置显式要求按验证 Top-1 选择 teacher checkpoint
- **THEN** teacher metrics MAY 使用最高验证 Top-1 epoch 作为 `best_epoch`
- **AND** teacher metrics MUST 将 `selection_metric` 记录为 Top-1 accuracy
- **AND** teacher metrics MUST 将 checkpoint 路径记录为 `best_top1.pth` 或其归档副本

#### Scenario: metrics 与 checkpoint 模态不一致
- **WHEN** teacher metrics 中的 `modality` 与 registry 当前模态不一致
- **THEN** registry 构建流程 MUST 拒绝该输入
- **AND** 错误信息 MUST 包含期望模态和实际模态

## ADDED Requirements

### Requirement: Teacher reliability registry checkpoint objective selection
Teacher reliability registry MUST 按 metrics 或 checkpoint metadata 中声明的 selection objective 选择 checkpoint。未显式要求 Top-1 teacher 时，registry MUST 优先使用 objective-selected checkpoint；对 LiDAR teacher，registry MUST NOT 在存在 `best.pth` 或 objective checkpoint metadata 时默认选择 `best_top1.pth`。

#### Scenario: registry 使用 metrics 中声明的 checkpoint
- **WHEN** teacher metrics 或 checkpoint sidecar metadata 提供可访问的 checkpoint 路径和 selection metadata
- **THEN** teacher reliability registry MUST 使用该 checkpoint 路径
- **AND** registry MUST 记录 `selection_metric`、`selection_mode`、`selected_epoch` 和 checkpoint 来源

#### Scenario: LiDAR teacher 默认使用 objective checkpoint
- **WHEN** LiDAR teacher run 同时包含 `checkpoints/best.pth` 和 `checkpoints/best_top1.pth`，且用户未显式要求 Top-1 teacher
- **THEN** teacher reliability registry MUST 选择 `checkpoints/best.pth`
- **AND** registry MUST NOT 因为 `best_top1.pth` 存在而覆盖 `best.pth`

#### Scenario: 显式 Top-1 teacher 使用 best_top1 checkpoint
- **WHEN** 用户显式指定 checkpoint 路径为 `best_top1.pth` 或 registry selection metric 为验证 Top-1
- **THEN** teacher reliability registry MAY 选择 `best_top1.pth`
- **AND** registry MUST 将 checkpoint 来源标记为 explicit 或 top1-selection

#### Scenario: objective checkpoint 缺失时报错
- **WHEN** LiDAR teacher run 缺少 metrics checkpoint 路径和 `checkpoints/best.pth`，且用户未显式要求 Top-1 teacher
- **THEN** registry 构建流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 提示提供 objective checkpoint、重建 teacher metrics，或显式选择 Top-1 checkpoint

#### Scenario: registry 保留 Top-1 诊断指标
- **WHEN** registry 使用 objective checkpoint 而 run 同时存在 Top-1 指标
- **THEN** teacher reliability registry MUST 保留 objective checkpoint 的指标
- **AND** registry MAY 记录最高 Top-1 epoch 和 Top-1 value 作为诊断字段
- **AND** Stage 2/3 默认 teacher 加载 MUST 使用 objective checkpoint 路径
