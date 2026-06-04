## MODIFIED Requirements

### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、primary 模型、supervised/adaptation loss、训练超参数、优化器、调度器、输出目录、随机种子、GPS-Rel-Polar 特征模式和 fusion 模态选择。配置 MUST 不再覆盖 KD 模式或 teacher checkpoint。

#### Scenario: 使用配置启动单模态训练
- **WHEN** 用户通过 CLI 传入 image、radar、GPS、LiDAR 或 mmWave 单模态训练配置
- **THEN** 系统 MUST 构建对应 dataset、primary model、loss、optimizer 和 scheduler
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 使用配置启动 fusion 训练
- **WHEN** 用户通过 CLI 传入 fusion 训练配置
- **THEN** 系统 MUST 构建同时包含启用模态输入的 dataset、fusion primary model、loss、optimizer 和 scheduler
- **AND** 系统 MUST 不要求 teacher checkpoint

### Requirement: 统一实验输出
训练和评估流程 MUST 将运行产物写入统一输出目录。输出目录 MUST 至少包含本次运行的有效配置、checkpoint 或权重引用、metrics、训练曲线或日志，以及测试报告。训练流程 MUST 记录 supervised/adaptation loss、训练准确率、验证指标和学习率；新产物 MUST 不记录蒸馏损失。

#### Scenario: 训练完成后保存进度日志
- **WHEN** 一次训练任务完成至少一个 epoch
- **THEN** 系统 MUST 在当前运行目录的训练日志中保存 epoch 级进度摘要
- **AND** 进度摘要 MUST 包含 epoch 编号、训练损失、训练主任务损失、训练准确率、验证损失、验证准确率和学习率
- **AND** 进度摘要 MUST 不包含新的训练蒸馏损失字段

#### Scenario: TensorBoard 不写蒸馏标量
- **WHEN** 一次训练 epoch 完成且 TensorBoard 日志启用
- **THEN** event 文件 MUST 记录训练总损失、训练主任务损失、训练准确率、验证损失、验证准确率、学习率和当前评估指标
- **AND** event 文件 MUST 不新增 `loss/distillation` 或 KD 标量

## REMOVED Requirements

### Requirement: Radar-only KD 实验配置
**Reason**: radar-only KD 配置随蒸馏支持删除。
**Migration**: 使用 `configs/radar/strong.yaml` 或 `configs/radar/lightweight.yaml`。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入旧 radar logits KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen `radar_teacher`

### Requirement: Legacy KD 配置生命周期可审计
**Reason**: 仓库不再保留可运行 legacy KD 配置，因此不需要维护 optional baseline lifecycle。
**Migration**: 历史结果只读保留；新实验使用 supervised/adaptation metadata。

#### Scenario: KD 配置带生命周期标记
- **WHEN** 仓库中出现 `configs/**/logits_kd.yaml`、`configs/**/rkd.yaml` 或等价 KD recipe
- **THEN** 检查 MUST 失败
- **AND** 开发者 MUST 删除该入口或重新提案新的蒸馏能力

### Requirement: 实验 summary 区分 KD 与 mainline
**Reason**: 新 summary 不再接收可运行 KD baseline。
**Migration**: 历史 summary 中的 KD 字段可只读展示；新 summary 使用 method family 和 training mode。

#### Scenario: mainline ranking 排除 legacy KD
- **WHEN** summary 读取新训练产物
- **THEN** 产物 MUST 不包含 legacy KD 分组
- **AND** 排名 MUST 基于当前 workflow eligibility 规则

