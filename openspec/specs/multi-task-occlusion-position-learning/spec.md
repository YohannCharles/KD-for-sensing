# multi-task-occlusion-position-learning Specification

## Purpose
定义 occlusion/position 辅助监督、多任务 loss、metrics 和 artifact 记录契约。
## Requirements
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
训练和评估流程 MUST 支持 beam supervised 主损失、遮挡 BCE 和位置 MSE 的加权组合，并 MUST 输出遮挡 accuracy、blocked-class F1 和位置 RMSE。无效标签位置 MUST 被 mask，不得参与 loss 或指标。

#### Scenario: supervised 多任务训练
- **WHEN** 多任务辅助监督启用且 batch/model 均提供辅助字段
- **THEN** 总 loss MUST 等于 beam supervised 基础 loss 加上配置权重后的遮挡 loss 和位置 loss
- **AND** 训练日志 MUST 分别记录 `loss/occlusion`、`loss/position` 和 `loss/multitask_total`
- **AND** 系统 MUST 不计算 KD 基础 loss

#### Scenario: 遮挡指标
- **WHEN** 验证或评估输出 `occlusion_logits` 且 batch 包含 `occlusion_label`
- **THEN** metrics MUST 包含遮挡 accuracy 和 blocked-class F1
- **AND** 计算 MUST 只使用 `occlusion_valid` 为真的位置

#### Scenario: 位置 RMSE
- **WHEN** 验证或评估输出 `position` 且 batch 包含 `position_target`
- **THEN** metrics MUST 包含米制 position RMSE
- **AND** 计算 MUST 只使用 `position_valid` 为真的位置

### Requirement: 训练流程支持多任务辅助 loss
训练流程 MUST 在保持现有 beam/KD 基础 loss 的前提下支持可选多任务辅助 loss。辅助 loss MUST 只在配置启用且 batch/model 均提供对应字段时计算；否则训练流程 MUST 保持现有 beam-only 行为。

#### Scenario: no-KD 多任务训练
- **WHEN** 用户运行启用遮挡和位置辅助任务的 no-KD fusion 训练
- **THEN** 训练流程 MUST 计算 beam CE、遮挡 BCE 和位置 MSE
- **AND** optimizer step MUST 使用三者加权后的总 loss
- **AND** train log MUST 记录每个 loss 分量

#### Scenario: KD 多任务训练
- **WHEN** 用户运行启用辅助任务的 logits KD 或 RKD fusion 训练
- **THEN** 训练流程 MUST 保留既有 KD 基础 loss 计算
- **AND** 训练流程 MUST 将辅助 loss 加到 student 总 loss
- **AND** teacher 模型 MUST 不被要求输出辅助头，除非配置显式启用 teacher auxiliary supervision

#### Scenario: 辅助字段缺失
- **WHEN** 配置启用辅助 loss 但 batch 或模型输出缺少必要字段
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的是 dataset target 还是 model auxiliary output

### Requirement: 验证和评估输出辅助指标
验证和评估流程 MUST 在启用多任务辅助监督时输出遮挡和位置指标，同时保留现有 Top-K、DBA、loss、degradation baseline 和 modality subset 评估语义。

#### Scenario: 验证输出遮挡指标
- **WHEN** 验证流程收到 `occlusion_logits` 和 `occlusion_label`
- **THEN** validation metrics MUST 包含遮挡 accuracy 和 blocked-class F1
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: 验证输出位置指标
- **WHEN** 验证流程收到 `position` 和 `position_target`
- **THEN** validation metrics MUST 包含 position RMSE
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: beam 指标保留
- **WHEN** 多任务辅助监督启用
- **THEN** 验证和评估流程 MUST 继续输出 beam Top-K、DBA、ATop-3、ATop-5 和 ADBA
- **AND** early stopping 默认 MUST 继续支持现有 `val_adba` 配置

### Requirement: 多任务运行产物可复现
训练和评估流程 MUST 在运行产物中记录多任务配置、遮挡阈值、辅助目标统计、loss 权重和辅助指标，确保后续评估和复现实验能加载相同的标签生成状态。

#### Scenario: final config 记录多任务状态
- **WHEN** 训练启用多任务辅助监督
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录遮挡阈值、阈值分位数、位置目标来源和 loss 权重
- **AND** checkpoint 或 normalization artifacts MUST 记录独立评估所需的辅助目标统计

#### Scenario: train log 记录辅助指标历史
- **WHEN** 训练至少完成一个 epoch 且启用多任务辅助监督
- **THEN** `train_log.json` MUST 包含遮挡和位置指标历史
- **AND** `training_outputs.npz` MUST 保存可画曲线的辅助 loss 或指标数组

### Requirement: Objective-aware 训练流程
训练流程 MUST 根据 `experiment.objective` 选择主 target、主模型输出、主 loss 和训练日志字段。`experiment.task` MUST 继续决定输入路由和模型 forward 路径。

#### Scenario: fusion occlusion 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: occlusion` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用遮挡 logits 和遮挡标签计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion position 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: position` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用位置输出和位置目标计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion multitask 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: multitask` 的训练配置
- **THEN** trainer MUST 计算 beam、occlusion 和 position 三个 loss 分量
- **AND** trainer MUST 按配置权重合成总 loss

### Requirement: Objective-aware 验证和评估
验证和评估流程 MUST 根据 `experiment.objective` 输出当前目标的主 metrics，并保留可计算的诊断 metrics。主 metrics MUST 支持 checkpoint 选择和 standalone evaluate。

#### Scenario: occlusion 验证指标
- **WHEN** 验证 `experiment.objective: occlusion` 的模型
- **THEN** validator MUST 输出遮挡 loss、accuracy 和 blocked-class F1
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_occlusion_blocked_f1`

#### Scenario: position 验证指标
- **WHEN** 验证 `experiment.objective: position` 的模型
- **THEN** validator MUST 输出位置 loss、RMSE 和 MAE
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_position_rmse`

#### Scenario: multitask 验证指标
- **WHEN** 验证 `experiment.objective: multitask` 的模型
- **THEN** validator MUST 输出 beam、occlusion 和 position 的分任务 metrics
- **AND** validator MUST 输出 multitask 总 loss 或配置指定的主指标

### Requirement: Objective-aware checkpoint registry
checkpoint registry 和 final config MUST 记录 objective-aware 指标，确保后续 evaluation 能按训练目标解释 checkpoint。

#### Scenario: 归档 occlusion checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: occlusion` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和遮挡指标
- **AND** evaluate MUST 能读取 registry artifact 并复用遮挡阈值

#### Scenario: 归档 position checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: position` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和位置指标
- **AND** evaluate MUST 能读取 registry artifact 并复用位置 target scaler

### Requirement: Fusion 多任务配置入口
Fusion 配置 MUST 能声明多任务辅助监督相关选项，包括启用状态、遮挡阈值分位数、位置目标来源、辅助 head 开关和 loss 权重。默认 fusion 配置 MUST 保持 beam-only，recommended 多任务配置 MUST 显式启用五模态和 auxiliary heads。

#### Scenario: 默认 fusion 配置保持 beam-only
- **WHEN** 用户加载现有 canonical fusion 配置
- **THEN** 配置 MUST 不默认启用遮挡或位置辅助任务
- **AND** 模型和 dataset MUST 保持现有 beam-only 行为

#### Scenario: 五模态多任务推荐配置
- **WHEN** 用户加载 recommended 五模态多任务 fusion 配置或 overlay
- **THEN** 配置 MUST 设置 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]`
- **AND** 配置 MUST 启用 `cls_token_transformer_fusion` 的遮挡和位置辅助头
- **AND** 配置 MUST 启用 dataset 的遮挡和位置目标生成

#### Scenario: loss 权重可配置
- **WHEN** 用户在 fusion 配置中设置 beam、遮挡或位置 loss 权重
- **THEN** 训练流程 MUST 使用配置值计算多任务总 loss
- **AND** final config MUST 记录实际使用的权重

### Requirement: Fusion 配置校验多任务依赖
系统 MUST 对多任务 fusion 配置进行显式校验。启用遮挡目标时必须能访问 beam sweep power 文件；启用位置目标时必须声明位置目标来源；启用 auxiliary loss 时模型必须支持对应辅助输出。

#### Scenario: 位置目标缺少来源
- **WHEN** 配置启用位置辅助任务但未声明合法 `position_target_source`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出支持的 position target source

#### Scenario: 模型不支持辅助输出
- **WHEN** 配置启用遮挡或位置 loss，但 `model.student` 不支持对应 auxiliary head
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出模型输出缺少的辅助字段

#### Scenario: 遮挡目标与数据文件不匹配
- **WHEN** 配置启用遮挡目标但监督 beam 文件不是 64 维 power vector
- **THEN** dataset 构建或首次取样 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出遮挡标签生成依赖 64-beam sweep

### Requirement: Fusion objective 配置矩阵
系统 MUST 为 fusion 实验提供 objective-aware 配置入口，使同一模态集合能够分别运行 `beam`、`occlusion`、`position` 和 `multitask` 预测目标。配置命名 MUST 同时表达模态集合和预测目标。

#### Scenario: 五模态 objective 配置
- **WHEN** 用户查看 recommended 五模态 fusion 配置
- **THEN** 系统 MUST 提供或虚拟解析 beam、occlusion、position 和 multitask 四类 objective 入口
- **AND** 每个入口 MUST 使用相同的五模态集合 `[image, radar, gps, lidar, mmwave]`

#### Scenario: 配置名表达 objective
- **WHEN** 用户使用 objective-aware fusion 配置
- **THEN** 配置名或 virtual config stem MUST 包含 canonical 模态 slug 和 objective 名称
- **AND** 配置中的 `experiment.objective` MUST 与名称中的 objective 一致

#### Scenario: 旧 no-KD 配置退役
- **WHEN** 用户继续使用已退役的 `configs/fusion/all_modalities_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该旧配置
- **AND** 错误信息 MUST 指向当前 supervised 或 objective-aware 入口

### Requirement: 模态失衡 objective 子集
fusion 配置系统 MUST 支持为模态失衡研究生成强模态、弱模态、单模态和全模态 objective 对照实验。每个 objective 配置 MUST 使用同一套 target 生成语义和同一套 metric 名称。

#### Scenario: strong-only occlusion 配置
- **WHEN** 用户请求 strong-only 模态集合的 occlusion fusion 配置
- **THEN** 系统 MUST 能生成只包含 strong modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: occlusion`

#### Scenario: weak-only position 配置
- **WHEN** 用户请求 weak-only 模态集合的 position fusion 配置
- **THEN** 系统 MUST 能生成只包含 weak modalities 的 fusion 配置
- **AND** 配置 MUST 设置 `experiment.objective: position`

#### Scenario: objective 间可比性
- **WHEN** 用户比较同一模态集合下的 beam、occlusion、position 和 multitask 结果
- **THEN** 系统 MUST 保持数据 split、target horizon、模态顺序和模型 backbone 默认配置一致

### Requirement: Objective-aware multitask canonical 默认等权
objective-aware fusion canonical 配置 MUST 在 `experiment.objective: multitask` 时默认使用 beam、occlusion 和 position 三个任务等权 loss。该默认值 MUST 应用于所有由 virtual canonical generator 生成的 multitask fusion 配置，包括 all-modalities、strong-only、weak-only 和显式模态 slug。

#### Scenario: 五模态 multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/image_radar_gps_lidar_mmwave_multitask_supervised.yaml`
- **THEN** 解析后的配置 MUST 设置 `experiment.objective: multitask`
- **AND** 解析后的配置 MUST 启用 beam、occlusion 和 position 三类 targets 与 heads
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.beam: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.occlusion: 1.0`
- **AND** 解析后的配置 MUST 设置 `loss.objective.weights.position: 1.0`

#### Scenario: strong-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/strong_only_multitask_supervised.yaml`
- **THEN** 解析后的配置 MUST 只包含 strong modalities `[gps, mmwave]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: weak-only multitask 默认等权
- **WHEN** 开发者加载 `configs/fusion/weak_only_multitask_supervised.yaml`
- **THEN** 解析后的配置 MUST 只包含 weak modalities `[image, radar, lidar]`
- **AND** 解析后的配置 MUST 设置 beam、occlusion 和 position 三个 objective 权重均为 `1.0`

#### Scenario: 显式 multitask 权重覆盖
- **WHEN** 用户通过实体 YAML 或命令行覆盖显式设置 `loss.objective.weights.position`
- **THEN** 系统 MUST 使用用户显式配置的 position 权重
- **AND** 该覆盖 MUST 不改变未被覆盖的 beam 和 occlusion 权重

#### Scenario: multitask 权重记录到产物
- **WHEN** 完成 objective-aware multitask 训练
- **THEN** `final_config.yaml` 或等价 runtime metadata MUST 能追溯 beam、occlusion 和 position 的实际 loss 权重
- **AND** epoch log MUST 记录或能派生本次 multitask 总 loss 的权重组成
