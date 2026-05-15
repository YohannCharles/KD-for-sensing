## ADDED Requirements

### Requirement: 多任务目标按需加载
DeepSense6G dataset MUST 在配置启用多任务辅助监督时按需生成并返回遮挡和位置目标。未启用多任务辅助监督时，dataset MUST 不读取 future GPS target，不拟合遮挡阈值，且不得改变现有按模态懒加载行为。

#### Scenario: beam-only 不读取辅助目标
- **WHEN** 用户运行未启用多任务辅助监督的 image、radar、GPS、LiDAR、mmWave 或 fusion 配置
- **THEN** dataset MUST 不返回 `occlusion_label`、`position_target`、`occlusion_valid` 或 `position_valid`
- **AND** dataset MUST 不读取 future GPS target 列
- **AND** dataset MUST 不扫描 beam power 文件拟合遮挡阈值

#### Scenario: 启用遮挡目标
- **WHEN** dataset 配置启用 `occlusion_target.enabled: true`
- **THEN** dataset MUST 使用 `future_beam` 路径对应的 64-beam power vector 生成每个预测时隙的 `occlusion_label`
- **AND** 返回样本 MUST 包含形状 `[num_pred]` 的 `occlusion_label` 和 `occlusion_valid`

#### Scenario: 启用位置目标
- **WHEN** dataset 配置启用 `position_target.enabled: true`
- **THEN** dataset MUST 返回形状 `[num_pred, 2]` 的 `position_target`
- **AND** 返回样本 MUST 包含形状 `[num_pred]` 的 `position_valid`

### Requirement: 辅助目标 artifact 复用
数据构建流程 MUST 将训练 split 拟合出的遮挡阈值和位置目标归一化统计作为运行 artifact 记录，并在测试 split 和独立评估中复用。测试 split MUST 不重新拟合这些统计量。

#### Scenario: 训练保存辅助目标统计
- **WHEN** 训练 dataset 启用遮挡目标或位置目标归一化
- **THEN** 训练流程 MUST 保存对应统计 artifact 到运行目录
- **AND** final config 或 run metadata MUST 记录 artifact 路径和关键统计值

#### Scenario: 测试复用训练统计
- **WHEN** 构建 test dataset 且辅助目标需要训练统计
- **THEN** 数据构建流程 MUST 将训练 dataset 的统计对象传给 test dataset
- **AND** test dataset MUST 不扫描测试集拟合阈值或 scaler

#### Scenario: 独立评估加载 artifact
- **WHEN** 用户运行评估入口并加载启用了多任务辅助监督的 checkpoint
- **THEN** 评估流程 MUST 从 checkpoint registry、normalization artifact 或显式配置加载遮挡阈值和位置统计
- **AND** 如果缺失必要 artifact，系统 MUST 抛出清晰错误

### Requirement: 序列预处理输出 future 位置列
序列 CSV 预处理 MUST 支持显式输出 future GPS/BS GPS 位置目标列。该开关启用时，每个合法窗口 MUST 在保留现有 `beam` 和 `future_beam` 列的同时，输出与预测 horizon 对齐的 `future_gps` 和 `future_bs_gps` 列。

#### Scenario: 生成 future GPS target 列
- **WHEN** 用户运行序列预处理并设置 `include_position_targets: true`
- **THEN** 输出 CSV MUST 包含 `future_gps1..future_gpsN`
- **AND** 输出 CSV MUST 包含 `future_bs_gps1..future_bs_gpsN`
- **AND** `N` MUST 等于配置的预测长度 `out_len`

#### Scenario: 不启用时保持旧 CSV 结构
- **WHEN** 用户运行序列预处理但未启用 `include_position_targets`
- **THEN** 输出 CSV MUST 保持现有列结构兼容
- **AND** 旧的 beam-only dataset MUST 能继续读取该 CSV

