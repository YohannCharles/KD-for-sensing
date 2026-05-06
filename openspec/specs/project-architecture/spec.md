# project-architecture Specification

## Purpose
Define the package-level architecture, lightweight import boundaries, responsibility splits for training/data/diagnostics modules, and the separation between source-controlled inputs and local runtime artifacts.
## Requirements
### Requirement: 可导入包结构
项目 MUST 提供 `src/kd_sensing/` Python 包，并将数据、模型、蒸馏、训练引擎、评估、预处理和通用工具放入职责清晰的子模块。包内模块 MUST 使用包内绝对导入或明确相对导入，不得依赖仓库根目录脚本名作为运行时导入条件。

#### Scenario: 从项目根目录导入包
- **WHEN** 开发者在项目根目录安装或设置本地包路径后执行 `import kd_sensing`
- **THEN** 导入 MUST 成功，并且不触发数据集读取、模型权重加载或训练逻辑

#### Scenario: 导入核心子模块
- **WHEN** 开发者导入 `kd_sensing.models`、`kd_sensing.data`、`kd_sensing.distillation`、`kd_sensing.engine` 和 `kd_sensing.preprocessing`
- **THEN** 每个子模块 MUST 成功导入，并暴露对应领域的公共构建入口或注册入口

### Requirement: 模块边界清晰
项目 MUST 按职责拆分当前根目录代码：模型定义进入 `models/`，数据集与样本解析进入 `data/`，KD 与 loss 进入 `distillation/` 或 `losses/`，训练/验证/测试循环进入 `engine/`，雷达和 CSV 预处理进入 `preprocessing/`，指标与 checkpoint 等通用逻辑进入 `utils/` 或 `evaluation/`。

#### Scenario: 新增模型时不修改数据模块
- **WHEN** 开发者新增一个 student 或 teacher 模型实现
- **THEN** 变更 MUST 限定在模型相关模块和注册代码内，且不需要修改 dataset、preprocess 或训练循环中的数据解析逻辑

#### Scenario: 新增预处理流程时不修改模型模块
- **WHEN** 开发者新增一种雷达或 CSV 预处理流程
- **THEN** 变更 MUST 限定在 `preprocessing/` 与配置/注册代码内，且不需要修改模型定义文件

### Requirement: 统一新脚本入口并移除旧入口
项目 MUST 使用 `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 或包内 CLI 作为唯一运行入口。项目 MUST 删除现有顶层旧脚本入口，包括 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 和 `gen_data_seq.py`，不得保留兼容包装脚本。

#### Scenario: 运行新训练脚本帮助信息
- **WHEN** 开发者执行 `python scripts/train.py --help`
- **THEN** 命令 MUST 正常退出，并展示配置文件、训练任务和命令行覆盖相关的参数说明

#### Scenario: 运行新评估和预处理脚本帮助信息
- **WHEN** 开发者执行 `python scripts/evaluate.py --help` 或 `python scripts/preprocess.py --help`
- **THEN** 命令 MUST 正常退出，并展示对应任务的参数说明

#### Scenario: 旧脚本入口已删除
- **WHEN** 结构重构完成后检查仓库根目录
- **THEN** 根目录 MUST 不存在 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 或 `gen_data_seq.py`

### Requirement: 项目根路径与资源路径统一
项目 MUST 提供统一路径解析工具，用于解析项目根目录、数据目录、权重目录、配置目录和输出目录。运行入口 MUST 通过该工具解析相对路径，避免依赖当前工作目录偶然匹配。

#### Scenario: 从子目录运行命令
- **WHEN** 开发者从仓库子目录调用新脚本或新 CLI，并传入相对数据路径
- **THEN** 系统 MUST 根据项目根路径解析资源位置，而不是错误地相对于当前子目录查找

#### Scenario: 读取默认资源目录
- **WHEN** 用户未显式传入数据目录或权重目录
- **THEN** 系统 MUST 使用配置中的默认路径，并能定位当前仓库的 `dataset/` 和 `All_models/` 目录

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

