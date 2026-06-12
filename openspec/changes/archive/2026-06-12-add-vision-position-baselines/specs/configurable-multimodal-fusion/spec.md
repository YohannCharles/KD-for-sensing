## ADDED Requirements

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
