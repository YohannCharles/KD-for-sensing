## ADDED Requirements

### Requirement: 多任务辅助监督配置
系统 MUST 提供可配置的多任务辅助监督开关，用于同时启用或关闭遮挡检测目标、位置估算目标、辅助 head、损失权重和评估指标。默认配置 MUST 保持 beam-only 行为，不读取辅助标签、不创建辅助 head、不改变主 loss。

#### Scenario: 默认 beam-only 行为
- **WHEN** 配置未启用多任务辅助监督
- **THEN** dataset MUST 只返回既有输入模态、`input_beam` 和 `target_beam`
- **AND** 模型 MUST 只要求主 beam logits 参与训练和评估
- **AND** 训练、验证、评估指标 MUST 与现有 beam-only 路径兼容

#### Scenario: 启用遮挡和位置辅助任务
- **WHEN** 配置启用 `occlusion` 和 `position` 辅助任务
- **THEN** dataset MUST 返回 `occlusion_label`、`occlusion_valid`、`position_target` 和 `position_valid`
- **AND** 支持该能力的模型 MUST 返回 `occlusion_logits` 和 `position`
- **AND** 训练流程 MUST 将 beam、遮挡和位置 loss 按配置权重合成为总 loss

### Requirement: 遮挡标签自动生成
系统 MUST 能从 beam sweep power vector 自动生成遮挡标签。对每个预测时隙，系统 MUST 读取对应监督 sweep，计算最大接收功率 `p_max`，并使用训练 split 上 `p_max` 的固定分位数阈值 `tau` 生成二值标签：`p_max < tau` 为遮挡，`p_max >= tau` 为非遮挡。

#### Scenario: 从训练 split 拟合遮挡阈值
- **WHEN** 训练 dataset 启用遮挡标签且未显式提供阈值
- **THEN** 系统 MUST 只扫描训练 split 的监督 sweep power 文件拟合 `tau`
- **AND** 系统 MUST 在运行 metadata 中记录 `threshold_percentile`、`tau`、样本数和正类比例

#### Scenario: 评估复用训练阈值
- **WHEN** 测试或评估 dataset 启用遮挡标签
- **THEN** 系统 MUST 使用训练 split 拟合或 checkpoint artifact 中记录的 `tau`
- **AND** 系统 MUST 不在测试 split 上重新拟合遮挡阈值

#### Scenario: 缺失 power 文件报错
- **WHEN** 启用遮挡标签但目标 beam power 文件不存在、为空或维度不是 64
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含出错路径和期望的 64-beam power vector 契约

### Requirement: 位置目标生成
系统 MUST 能为位置估算头生成二维本地平面位置目标。正式多任务实验 MUST 使用与预测 horizon 对齐的 future GPS/BS GPS 路径，读取 UE 和 BS 经纬度，转换为米制本地坐标，并返回 `ue_xy - bs_xy` 的 `[x, y]` 目标。

#### Scenario: future GPS 位置目标
- **WHEN** 配置使用 `position_target_source: future_gps_local_xy`
- **THEN** dataset MUST 读取 `future_gps1..future_gpsH` 和 `future_bs_gps1..future_bs_gpsH`
- **AND** `position_target` MUST 具有形状 `[num_pred, 2]`
- **AND** `position_valid` MUST 标记每个 horizon 的目标是否可用

#### Scenario: 缺失 future GPS 列
- **WHEN** 启用位置目标但序列 CSV 缺少 required future GPS 或 future BS GPS 列
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 提示重新运行启用 position target 的序列预处理

#### Scenario: smoke test fallback
- **WHEN** 配置显式使用 `position_target_source: last_input_gps_local_xy`
- **THEN** dataset MAY 使用最后一个历史 GPS/BS GPS 生成所有 horizon 的位置目标
- **AND** 运行 metadata MUST 标记该目标来源为 fallback，不得伪装为正式 future target

### Requirement: 多任务模型输出契约
支持多任务辅助监督的模型 MUST 保持主 beam logits 输出契约，并通过 diagnostics 或等价 dict 字段提供辅助输出。`occlusion_logits` MUST 对齐预测 horizon，`position` MUST 对齐预测 horizon 和二维坐标。

#### Scenario: 辅助输出形状
- **WHEN** batch size 为 `B`、预测 horizon 为 `H`、beam 类别数为 `C`
- **THEN** 主 `logits` MUST 具有形状 `[B, H, C]`
- **AND** `occlusion_logits` MUST 具有形状 `[B, H]`
- **AND** `position` MUST 具有形状 `[B, H, 2]`

#### Scenario: 旧模型不启用辅助任务
- **WHEN** 配置未启用多任务辅助监督或模型未声明支持 auxiliary heads
- **THEN** 训练流程 MUST 不要求 `occlusion_logits` 或 `position`
- **AND** 旧模型的 beam-only forward MUST 继续可用

### Requirement: 多任务损失和指标
训练和评估流程 MUST 支持 beam CE/KD 主损失、遮挡 BCE 和位置 MSE 的加权组合，并 MUST 输出遮挡 accuracy、blocked-class F1 和位置 RMSE。无效标签位置 MUST 被 mask，不得参与 loss 或指标。

#### Scenario: 训练 loss 合成
- **WHEN** 多任务辅助监督启用且 batch/model 均提供辅助字段
- **THEN** 总 loss MUST 等于既有 beam/KD 基础 loss 加上配置权重后的遮挡 loss 和位置 loss
- **AND** 训练日志 MUST 分别记录 `loss/occlusion`、`loss/position` 和 `loss/multitask_total`

#### Scenario: 遮挡指标
- **WHEN** 验证或评估输出 `occlusion_logits` 且 batch 包含 `occlusion_label`
- **THEN** metrics MUST 包含遮挡 accuracy 和 blocked-class F1
- **AND** 计算 MUST 只使用 `occlusion_valid` 为真的位置

#### Scenario: 位置 RMSE
- **WHEN** 验证或评估输出 `position` 且 batch 包含 `position_target`
- **THEN** metrics MUST 包含米制 position RMSE
- **AND** 计算 MUST 只使用 `position_valid` 为真的位置

