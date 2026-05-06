## ADDED Requirements

### Requirement: 轻量导入边界
项目 MUST 区分轻量基础模块和重依赖运行模块。导入配置加载、路径解析、场景元数据和模态契约时，系统 MUST 不导入 dataset、model、diagnostics、训练循环或需要 pandas、scipy、skimage、matplotlib 的模块。

#### Scenario: 缺少数据依赖时加载配置模块
- **WHEN** Python 环境可导入 `kd_sensing` 但缺少 pandas、scipy、skimage 或 matplotlib 中任一数据/可视化依赖
- **THEN** `import kd_sensing.config` MUST 成功
- **AND** 该导入 MUST 不触发 dataset 类、模型类或诊断渲染模块导入

#### Scenario: 只导入路径工具
- **WHEN** 开发者执行 `from kd_sensing.utils.paths import resolve_path`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 checkpoint registry、dataset 或模型模块

#### Scenario: 组件构建时才导入默认组件
- **WHEN** 训练或评估构建 dataset、model、loss、metric、distiller 或 preprocessor
- **THEN** 系统 MUST 显式导入默认组件以完成注册
- **AND** 该默认组件导入边界 MUST 不影响轻量配置加载路径

### Requirement: 横切 builder 职责拆分
训练引擎 MUST 将配置到运行对象的构建逻辑按职责拆分。dataset/dataloader 构建、启用模态推导、cache policy、归一化 artifact、run metadata、optimizer/scheduler/device 构建 MUST 有明确模块边界。

#### Scenario: 修改 cache policy 不触碰 optimizer 构建
- **WHEN** 开发者调整 image 或 LiDAR cache policy 解析逻辑
- **THEN** 变更 MUST 限定在 cache policy 相关模块及其测试
- **AND** 不需要修改 optimizer、scheduler 或 device 构建逻辑

#### Scenario: 修改 normalization artifact 不触碰 dataset 模态解析
- **WHEN** 开发者调整 GPS、LiDAR 或 mmWave 归一化 artifact 的保存和加载格式
- **THEN** 变更 MUST 限定在归一化 artifact 相关模块及其测试
- **AND** 不需要修改启用模态推导逻辑

#### Scenario: 旧 builders import 兼容
- **WHEN** 现有代码从 `kd_sensing.engine.builders` 导入公开构建函数
- **THEN** 导入 MUST 继续成功
- **AND** 函数语义 MUST 与拆分前保持兼容

### Requirement: 模态数据转换职责拆分
数据转换模块 MUST 按 image、radar、lidar、gps、mmwave 和通用 IO/cache/normalization 职责组织。新增或修改某个模态的数据读取、特征构造或 cache key 时，变更 MUST 不要求编辑其它模态的转换实现。

#### Scenario: 修改 GPS 特征不触碰 LiDAR 转换
- **WHEN** 开发者修改 GPS feature sequence 构造
- **THEN** 变更 MUST 限定在 GPS 转换相关模块和测试
- **AND** 不需要修改 LiDAR BEV、image motion mask 或 mmWave feature 转换实现

#### Scenario: 兼容旧 transforms 导入
- **WHEN** 现有代码从 `kd_sensing.data.transforms` 导入已公开的转换函数或 scaler
- **THEN** 导入 MUST 继续成功
- **AND** 函数行为 MUST 与拆分前保持兼容

### Requirement: 诊断可视化内部模块化
诊断可视化实现 MUST 将配置解析、数据集准备、样本选择、统计汇总、渲染和文件写出拆成职责明确的内部模块。公开入口和 CLI 行为 MUST 保持兼容。

#### Scenario: 公开诊断入口兼容
- **WHEN** 现有代码调用 `kd_sensing.diagnostics.modality_visualization.visualize_modalities`
- **THEN** 调用 MUST 继续成功
- **AND** 返回 payload、summary 路径和主要输出文件语义 MUST 保持兼容

#### Scenario: 修改样本选择不触碰渲染逻辑
- **WHEN** 开发者调整按 `seq_index` 或 label 选择样本的策略
- **THEN** 变更 MUST 限定在 sampling 相关模块和测试
- **AND** 不需要修改 PNG 渲染函数或 metadata 写出函数

#### Scenario: 修改 PNG 渲染不触碰数据集构建
- **WHEN** 开发者调整单样本 PNG 布局
- **THEN** 变更 MUST 限定在 render 相关模块和可视化测试
- **AND** 不需要修改诊断 dataset 构建逻辑

### Requirement: 源码与实验产物边界
项目 MUST 明确源码、配置、文档、OpenSpec artifacts 与本地数据、训练日志、缓存和输出产物的边界。本地运行产物 MUST 保持在 `.gitignore` 覆盖范围内，文档 MUST 指明哪些目录是可复现输入、哪些目录是可删除生成物。

#### Scenario: 本地产物不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令
- **THEN** 生成的 logs、outputs、cache、checkpoint 和 Python bytecode 产物 MUST 位于忽略规则覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

#### Scenario: 文档说明产物边界
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `dataset/`、`All_models/`、`outputs/`、`logs/` 和 cache 目录的角色
- **AND** 文档 MUST 指明哪些目录通常不应纳入源码变更
