# vision-position-baseline-suite Specification

## Purpose
Define the current DeepSense6G vision-position baseline suite, including supported image+GPS and GPS-only presets, BeamBench-aligned configuration contracts, model registry requirements, evaluation metadata, and reporting caveats.
## Requirements
### Requirement: Vision-Position baseline preset 矩阵
系统 MUST 提供 DeepSense6G Vision-Position beam prediction baseline preset 矩阵，至少覆盖 Camera AE + GPS、ResNet + GPS late fusion、Transformer image+gps fusion 和 GPS-only neural baseline。每个 preset MUST 通过现有配置加载和 `MODELS` registry 构建，不得要求新增绕过 `kd_sensing` 包结构的长期训练脚本。

#### Scenario: Camera AE + GPS preset 可构建
- **WHEN** 用户加载 Camera AE + GPS baseline 配置
- **THEN** 配置 MUST 设置 DeepSense6G dataset、启用 image 与 gps 输入、输出 64 类 beam logits
- **AND** 模型 MUST 使用 Camera AE encoder 或其 frozen encoder wrapper 提取图像 latent
- **AND** 模型 MUST 使用 GPS direct/MLP embedding 后与图像 latent 融合

#### Scenario: ResNet + GPS preset 可构建
- **WHEN** 用户加载 ResNet + GPS baseline 配置
- **THEN** 配置 MUST 启用 image 与 gps 输入
- **AND** 模型 MUST 使用可配置 ResNet 或等价视觉 backbone 生成帧级视觉特征
- **AND** 模型 MUST 将视觉特征、GPS embedding 和 temporal aggregation 输出连接到 64 类 classifier

#### Scenario: Transformer image+gps preset 可构建
- **WHEN** 用户加载 Transformer image+gps baseline 配置
- **THEN** 配置 MUST 启用 image 与 gps 输入
- **AND** 模型 MUST 将视觉特征和 GPS embedding 组织为带位置或类型信息的 token
- **AND** Transformer encoder 输出 MUST 连接到 64 类 beam classifier

#### Scenario: GPS-only neural preset 可构建
- **WHEN** 用户加载 GPS-only neural baseline 配置
- **THEN** 配置 MUST 只启用 GPS 输入和必要 beam label
- **AND** 模型 MUST 使用 MLP、GRU 或 LSTM 类神经网络处理 GPS 序列
- **AND** run metadata MUST 标记该 baseline 使用神经网络，而不是已有非神经 GPS window baseline
- **AND** metadata MUST 标记该 preset 不等价于 Arnold22 BeamBench Table III 的 GPS `Classical*` 或 `Dense†` 行

### Requirement: 统一输入输出契约
Vision-Position baseline 模型 MUST 接收现有 DeepSense6G batch 字段，并保持统一 logits 输出契约。图像输入 MUST 支持 `[B, T, C, H, W]`，GPS 输入 MUST 支持 `[B, T, D]`，输出 logits MUST 支持 `[B, H, 64]` 或可由现有 engine 标准化为该形状，其中 `H` 为预测 horizon。

#### Scenario: 图像和 GPS 序列维度对齐
- **WHEN** image+gps baseline forward 接收 batch
- **THEN** 模型 MUST 校验 image 和 GPS 的 batch 维与 sequence 维一致
- **AND** 若维度不一致，系统 MUST 抛出清晰错误而不是静默广播或截断

#### Scenario: 单 horizon 输出
- **WHEN** 配置设置 `num_pred: 1`
- **THEN** baseline logits MUST 能被训练和评估流程解释为 `[B, 1, 64]`
- **AND** top-k metrics MUST 针对该 horizon 计算

#### Scenario: 多 horizon 输出
- **WHEN** 配置设置 `num_pred` 大于 1
- **THEN** baseline logits MUST 覆盖每个预测 horizon
- **AND** 训练标签和评估指标 MUST 不把历史输入 beam 拼入未来目标

### Requirement: 可配置的预处理、归一化和增强
Vision-Position baseline 配置 MUST 明确 image profile、image normalization、可选 augmentation、GPS feature mode 和 GPS 归一化策略。默认 image+gps preset MUST 不读取未启用的 radar、LiDAR、mmWave 或 CSI 输入。

#### Scenario: ResNet preset 使用 ImageNet profile
- **WHEN** 用户选择 ResNet ImageNet baseline
- **THEN** 配置 MUST 声明 RGB/ImageNet image profile 和输入尺寸
- **AND** encoder MUST 在输入通道或尺寸不符合要求时给出清晰错误

#### Scenario: GPS 归一化只使用训练统计
- **WHEN** baseline 需要 fit GPS normalization artifact
- **THEN** 系统 MUST 只使用训练 split 或合法 calibration split 估计统计量
- **AND** validation/test split MUST 复用训练统计，不得重新 fit

#### Scenario: augmentation 可关闭
- **WHEN** 用户运行 deterministic smoke 或评估配置
- **THEN** image augmentation MUST 可通过配置关闭
- **AND** run metadata MUST 记录 augmentation 是否启用

### Requirement: 训练评估闭环和 top-k 指标
Vision-Position baseline preset MUST 能通过现有 `kd-sensing-train` 和 `kd-sensing-evaluate` 工作流完成 supervised 训练和评估闭环。评估输出 MUST 至少包含 top-1 和 top-3 accuracy，并在可用时包含现有 BeamBench DBA 或 circular beam metric 字段。

#### Scenario: 训练保存 best checkpoint
- **WHEN** 用户运行任一 baseline 训练配置
- **THEN** 系统 MUST 完成 forward、loss、backward、optimizer step、validation metric 和 checkpoint 保存
- **AND** best checkpoint 选择指标 MUST 写入 run metadata 或 train log

#### Scenario: 评估 top-k 指标
- **WHEN** 用户运行任一 baseline 评估配置
- **THEN** 系统 MUST 计算 top-1 和 top-3 accuracy
- **AND** 若输出 top-2 或 top-5，字段名 MUST 清晰区分对应 k 值

#### Scenario: 指标口径可审计
- **WHEN** baseline 输出 metrics artifact
- **THEN** artifact MUST 记录 label space、beam shift、metric profile 和是否使用 circular beam distance
- **AND** 不同口径的 DBA 或 normalized gain 字段 MUST 不混用同一字段名
- **AND** BeamBench-aligned preset MUST 记录 `beam_target_source`，并在 Table III 口径下使用当前 beam target 而不是 sequence `future_beam*` target

### Requirement: baseline metadata 和本地产物边界
Vision-Position baseline run MUST 记录足以复现实验配置的 metadata，并遵守本仓库本地产物边界。metadata MUST 包含 baseline preset、启用模态、encoder 类型、GPS feature mode、temporal aggregation、num classes、num_pred、image profile、normalization artifact 和 mock/real data 标记。

#### Scenario: metadata 记录 baseline preset
- **WHEN** baseline 训练或评估完成
- **THEN** run metadata MUST 包含 `baseline_preset`
- **AND** metadata MUST 区分 `camera_ae_gps`、`resnet_gps`、`transformer_image_gps` 和 `gps_only_neural`

#### Scenario: 本地产物不提交
- **WHEN** baseline 运行生成 checkpoint、predictions、TensorBoard、cache 或 report
- **THEN** 这些文件 MUST 位于 ignored 的 `outputs/`、`logs/` 或其它已声明本地产物目录
- **AND** 源码变更 MUST NOT 要求提交真实数据、新生成 checkpoint 或训练日志

#### Scenario: mock 结果标记
- **WHEN** baseline 在 mock dataset 或 synthetic smoke 数据上运行
- **THEN** metrics、checkpoint metadata 和 report MUST 标记 `mock_data: true`
- **AND** mock metrics MUST NOT 被描述为真实 DeepSense6G 或官方 BeamBench 结果

### Requirement: baseline smoke 和回归测试
Vision-Position baseline suite MUST 提供不依赖真实 DeepSense6G 数据的快速测试覆盖，用于验证配置加载、模型构建、forward shape、top-k metrics 和 DataLoader 字段选择。所有项目相关 Python 测试命令 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: 模型 forward smoke
- **WHEN** 测试使用小型随机 image 与 GPS batch 构建每个 baseline
- **THEN** 每个模型 MUST 完成 forward
- **AND** 输出 logits 的 batch、horizon 和 class 维度 MUST 符合配置

#### Scenario: 配置加载 smoke
- **WHEN** 测试加载四类 baseline preset 配置
- **THEN** 配置加载 MUST 成功
- **AND** 启用模态 MUST 与 preset 声明一致

#### Scenario: CLI help 仍可用
- **WHEN** 开发者运行 `conda run -n kd_mm_beam kd-sensing-train --help` 或 `conda run -n kd_mm_beam kd-sensing-evaluate --help`
- **THEN** 命令 MUST 正常退出
- **AND** 新 baseline 不得破坏现有训练和评估入口

### Requirement: image+gps fusion baseline preset 选择
Configurable fusion MUST 支持 image+gps baseline preset 在 late-concat fusion 和 transformer token fusion 之间选择 primary model。两类 preset MUST 复用现有 fusion 模态选择、batch 准备、supervised loss、checkpoint 和评估流程，并保持未启用模态不被读取。

#### Scenario: image+gps late-concat fusion preset
- **WHEN** 用户加载 image+gps late-concat baseline 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** primary model modalities MUST 等于 `["image", "gps"]`
- **AND** primary model MUST 将 image encoder 输出与 GPS encoder 输出融合后预测 64 类 logits

#### Scenario: image+gps transformer fusion preset
- **WHEN** 用户加载 image+gps transformer baseline 配置
- **THEN** 配置 MUST 设置 `experiment.task: fusion`
- **AND** primary model MUST 使用 transformer encoder 融合 image token 和 GPS token
- **AND** 配置 MUST 记录 `d_model`、`num_heads`、`num_layers`、dropout 和 max sequence length

#### Scenario: 未启用模态不被要求
- **WHEN** 用户运行 image+gps baseline preset
- **THEN** dataset 和 batch 准备 MUST 只要求 image、gps、input beam 和 target beam 字段
- **AND** radar、LiDAR、mmWave 或 CSI 文件缺失不得阻止该 preset 运行

#### Scenario: canonical 配置与显式实体 YAML 语义一致
- **WHEN** image+gps baseline preset 由 virtual config recipe 生成或由实体 YAML 加载
- **THEN** 两种来源 MUST 产生等价的 enabled modalities、model primary type、num classes、num_pred 和 dataset field 语义
- **AND** 训练和评估入口 MUST 不因配置来源不同而改变 batch 输入契约

### Requirement: image+gps fusion encoder 可替换
image+gps fusion baseline MUST 支持通过配置选择视觉 encoder 和 GPS encoder，而不要求修改 dataset 或训练循环。视觉 encoder 至少 MUST 支持 Camera AE encoder 和 ResNet ImageNet encoder；GPS encoder MUST 支持 direct MLP embedding 或现有 GPS feature extractor 风格。

#### Scenario: 切换视觉 encoder 不改数据模块
- **WHEN** 用户将 image+gps fusion preset 的视觉 encoder 从 Camera AE 切换为 ResNet
- **THEN** 变更 MUST 限定在模型配置和模型构建逻辑
- **AND** dataset MUST 继续提供同一 image batch 字段

#### Scenario: encoder metadata 写入 run
- **WHEN** image+gps fusion baseline 训练或评估完成
- **THEN** run metadata MUST 记录视觉 encoder 类型、是否使用 pretrained 权重、冻结策略和 GPS encoder 类型
- **AND** 若使用 ResNet pretrained 权重，metadata MUST 记录权重来源或配置值

#### Scenario: encoder 输入 profile 校验
- **WHEN** 用户配置的 image profile 与所选视觉 encoder 不兼容
- **THEN** 系统 MUST 在模型构建或首个 forward 前抛出清晰错误
- **AND** 错误信息 MUST 包含所需 image profile 或输入 shape
