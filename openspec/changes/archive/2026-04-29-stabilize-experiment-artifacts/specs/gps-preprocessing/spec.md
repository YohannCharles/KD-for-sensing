## MODIFIED Requirements

### Requirement: GPS 训练集 scaler
系统 MUST 只使用训练集 GPS 特征 fit scaler，并将同一 scaler 应用于验证或测试 split。系统 MUST 不使用验证集或测试集统计量参与 GPS 特征归一化。训练流程 MUST 将 fit 后的 GPS scaler 作为可复用工件保存，并在评估同一 checkpoint 时优先加载训练时保存的 scaler。

#### Scenario: 训练集 fit scaler
- **WHEN** dataloader 构建训练 split 且启用 GPS 归一化
- **THEN** 系统 MUST 使用训练 split 的 GPS 特征 fit scaler
- **AND** 系统 MUST 将 fit 后的 scaler 保存在训练 dataset 实例或可复用对象中

#### Scenario: 训练保存 scaler 工件
- **WHEN** 启用 GPS 归一化的训练流程完成 dataloader 构建
- **THEN** 系统 MUST 将训练集 fit 后的 GPS scaler 保存到当前运行目录的稳定工件路径
- **AND** 训练日志或 registry metadata MUST 记录该 scaler 工件路径

#### Scenario: 测试集复用训练 scaler
- **WHEN** dataloader 构建测试 split 且启用 GPS 归一化
- **THEN** 系统 MUST 使用训练 split已 fit 或从训练工件加载的 scaler 转换测试 GPS 特征
- **AND** 系统 MUST 不在测试 split 上重新 fit scaler

#### Scenario: 评估从 checkpoint metadata 加载 scaler
- **WHEN** 评估入口加载的 checkpoint metadata 或 registry sidecar 记录了 GPS scaler 路径
- **THEN** 系统 MUST 加载该 scaler 并传递给测试 dataset
- **AND** 系统 MUST 不为了 GPS scaler 重新扫描训练 split

#### Scenario: 缺少 scaler 的 GPS 评估
- **WHEN** 评估入口直接构建启用 GPS 归一化的测试 dataset 且没有可用训练 scaler、scaler 文件或 registry metadata
- **THEN** 系统 MUST 抛出清晰错误，提示需要提供训练集 scaler 或使用训练 dataloader 构建流程

## ADDED Requirements

### Requirement: GPS 平滑窗口死配置移除
系统 MUST 不再将 `gps_smooth_window` 作为受支持的 GPS 配置能力暴露。GPS `relative_polar` 特征构造 MUST 不依赖该字段，默认配置、示例配置、公开文档和显式参数管线 MUST 移除该字段。

#### Scenario: 默认配置不暴露 gps_smooth_window
- **WHEN** 用户查看默认 GPS 或包含 GPS 的实验配置
- **THEN** 配置中 MUST 不包含 `gps_smooth_window`
- **AND** README 或实验说明 MUST 不把 `gps_smooth_window` 描述为可用 GPS 特征参数

#### Scenario: GPS 特征构造不接收平滑窗口参数
- **WHEN** 系统构造 `relative_polar` GPS 特征
- **THEN** `build_gps_features` 和 `load_gps_feature_sequence` 的公开调用路径 MUST 不要求或传递 `gps_smooth_window`
- **AND** 输出 GPS 特征 MUST 仍保持 `[seq_len, 3]` 的 `[dist, sin_theta, cos_theta]` 语义

#### Scenario: 历史配置字段不改变特征
- **WHEN** 历史外部配置仍包含 `gps_smooth_window`
- **THEN** 系统 MUST 忽略该遗留字段或抛出包含迁移说明的清晰错误
- **AND** 系统 MUST 不因为该字段改变 GPS `relative_polar` 特征值
