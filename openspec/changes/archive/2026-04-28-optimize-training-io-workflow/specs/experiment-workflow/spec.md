## MODIFIED Requirements

### Requirement: 统一实验输出
训练和评估流程 MUST 将运行产物写入统一输出目录。输出目录 MUST 至少包含本次运行的有效配置、checkpoint 或权重引用、metrics、训练曲线或日志，以及测试报告。训练和评估流程 MUST 默认创建互不覆盖的运行目录；只有在用户显式启用覆盖、显式恢复训练或传入确定性输出目录时，系统才 MAY 复用既有目录。训练流程 MUST 在启用 TensorBoard 时写入可由 TensorBoard 读取的标量 event 日志，并且 MUST 支持通过配置关闭该日志写入。训练流程 MUST 在启用进度显示时提供 `tqdm` 训练进度条，并且 MUST 将每个 epoch 的进度摘要保存到运行日志。

#### Scenario: 训练完成后保存运行配置
- **WHEN** 一次训练任务启动并创建输出目录
- **THEN** 系统 MUST 保存解析和覆盖后的最终配置，便于后续复现实验

#### Scenario: 固定 run_name 默认不覆盖旧实验
- **WHEN** 用户设置 `output.run_name` 且目标运行目录已存在，并且未启用覆盖或 resume
- **THEN** 系统 MUST 创建带唯一 run id、时间戳或等价后缀的新运行目录
- **AND** 系统 MUST 不覆盖既有 `final_config.yaml`、`metrics.json`、checkpoint、TensorBoard event 或 `train_log.json`

#### Scenario: 显式恢复训练复用运行目录
- **WHEN** 用户设置 `training.resume: true` 且提供固定 `output.run_name`
- **THEN** 系统 MUST 从该运行目录下的 checkpoint 恢复训练
- **AND** 系统 MUST 不自动追加新的 run id

#### Scenario: 显式覆盖运行目录
- **WHEN** 用户显式启用输出覆盖配置并设置 `output.run_name`
- **THEN** 系统 MAY 复用该运行目录
- **AND** 系统 MUST 在最终配置或运行日志中记录覆盖行为

#### Scenario: 训练过程中显示 tqdm 进度
- **WHEN** 一次训练任务启动且进度显示配置启用
- **THEN** 系统 MUST 使用 `tqdm` 展示 epoch 或 batch 级训练进度
- **AND** 进度条 MUST 展示当前 epoch、batch 进度、训练损失、任务损失、蒸馏损失、训练准确率和学习率中的关键状态

#### Scenario: 训练完成后保存进度日志
- **WHEN** 一次训练任务完成至少一个 epoch
- **THEN** 系统 MUST 在当前运行目录的训练日志中保存 epoch 级进度摘要
- **AND** 进度摘要 MUST 包含 epoch 编号、训练损失、训练任务损失、训练蒸馏损失、训练准确率、验证损失、验证准确率和学习率
- **AND** 日志保存 MUST 保持既有历史指标数组兼容

#### Scenario: 通过配置关闭 tqdm 进度显示
- **WHEN** 用户在训练配置中关闭进度显示
- **THEN** 系统 MUST 不创建可视化 `tqdm` 进度条
- **AND** 系统 MUST 继续保存训练日志和 epoch 级进度摘要

#### Scenario: 训练过程中写入 TensorBoard 标量日志
- **WHEN** 一次训练任务完成至少一个 epoch 且 TensorBoard 日志启用
- **THEN** 系统 MUST 在当前运行目录下写入 TensorBoard event 文件
- **AND** event 文件 MUST 记录训练总损失、训练任务损失、训练蒸馏损失、训练准确率、验证损失、验证准确率、学习率、验证 `ATop-3`、验证 `ATop-5` 和验证 `ADBA` 标量

#### Scenario: TensorBoard 记录跨时隙平均验证指标
- **WHEN** 一次训练 epoch 的验证阶段产出 per-slot Top-K accuracy 和 DBA 结果
- **THEN** 系统 MUST 将 `ATop-3` 计算为所有 `J + 1` 个目标时隙 Top-3 accuracy 的平均值
- **AND** 系统 MUST 将 `ATop-5` 计算为所有 `J + 1` 个目标时隙 Top-5 accuracy 的平均值
- **AND** 系统 MUST 将 `ADBA` 计算为所有 `J + 1` 个目标时隙 DBA 的平均值，且 DBA MUST 使用 Top-3 预测 beam 计算

#### Scenario: 通过配置关闭 TensorBoard 日志
- **WHEN** 用户在训练配置中关闭 TensorBoard 日志
- **THEN** 系统 MUST 不创建新的 TensorBoard writer，也不得因未写入 event 文件而影响训练完成

#### Scenario: 评估完成后保存指标
- **WHEN** 一次评估任务完成
- **THEN** 系统 MUST 在输出目录保存 Top-K、DBA、loss、latency 或当前评估入口支持的指标结果

#### Scenario: 评估默认不覆盖旧报告
- **WHEN** 用户多次运行评估入口且未显式指定覆盖同一输出目录
- **THEN** 系统 MUST 为每次评估创建互不覆盖的评估运行目录
- **AND** 系统 MUST 不固定覆盖 `outputs/evaluation/test_report.json`

#### Scenario: 输出记录 split 和样本数
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** 系统 MUST 在最终配置、运行日志或测试报告中记录实际使用的 train/test CSV 路径
- **AND** 系统 MUST 记录每个 split 的样本数，便于判断不同实验是否可横向比较

## ADDED Requirements

### Requirement: 跨模态可比较 split 配置
项目 MUST 提供可用于单模态和多模态横向比较的统一 split 配置方式。默认和 canonical 实验配置 MUST 让 image、radar、GPS、LiDAR 和 fusion 实验引用同一组 train/test CSV。

#### Scenario: 使用统一 split 运行单模态实验
- **WHEN** 用户将 image、radar、GPS 和 LiDAR 单模态配置指向同一组 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建各模态 dataset
- **AND** 训练或评估输出 MUST 记录相同的 CSV 路径和样本数

#### Scenario: 使用统一 split 运行 fusion 实验
- **WHEN** 用户将 fusion 配置指向与单模态相同的 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建 fusion dataset
- **AND** 未启用模态不得影响该 split 的可用性

#### Scenario: 默认配置使用统一 split
- **WHEN** 开发者查看默认 image、radar、GPS、LiDAR 和 fusion 实验配置
- **THEN** 每个配置 MUST 指向 `train_seqs_RA_GPS_LIDAR.csv` 和 `test_seqs_RA_GPS_LIDAR.csv`
- **AND** 输出 MUST 清晰记录该统一 split 的路径和样本数
