## ADDED Requirements

### Requirement: 训练方法扩展点边界
训练引擎 MUST 提供明确的训练方法扩展点，用于接入 G2D、CRAF、MARF 或后续类似方法的 teacher runtime、额外 loss、梯度后处理和 epoch diagnostics。`kd_sensing.engine.trainer` MUST 保持训练生命周期编排职责，不得继续作为方法特有 loss、teacher ensemble、counterfactual 或 subset-training 逻辑的主要实现位置。

#### Scenario: 新增训练方法不扩写主循环
- **WHEN** 开发者新增一个需要额外 loss 或 diagnostics 的训练方法
- **THEN** 主要实现 MUST 位于方法扩展模块及其测试中
- **AND** `kd_sensing.engine.trainer` 中的 epoch/batch 主循环 MUST 仅通过通用扩展点调用该方法

#### Scenario: 现有方法迁移到扩展模块
- **WHEN** 开发者查看 G2D、CRAF 或 MARF 的训练期接入逻辑
- **THEN** teacher runtime、extra loss、subset/counterfactual forward 和 scalar diagnostics 的主要实现 MUST 位于对应 engine 方法模块
- **AND** `trainer.py` MUST 不再包含这些方法的大段私有 helper 实现

### Requirement: 共享任务 forward runtime
训练、验证、诊断预测和 teacher runtime MUST 复用同一组任务 forward helper 来完成 batch 标准化、输入准备、model forward、输出适配和 future slot 选择。新增或修改模态输入准备、task forward 参数或强制模态 mask 行为时，变更 MUST 不要求在 trainer、validator、viewer prediction 和 teacher ensemble 中重复修改分支逻辑。

#### Scenario: 修改 fusion 输入准备只改 runtime helper
- **WHEN** 开发者调整 fusion task 的 `modalities` 输入准备或 force mask 透传逻辑
- **THEN** 主要变更 MUST 限定在共享 forward runtime 模块和测试
- **AND** 不需要分别修改 trainer、validator、viewer prediction 和 G2D teacher runtime 的 task 分支

#### Scenario: 验证路径复用训练输入契约
- **WHEN** 训练和验证使用同一个 fusion 配置运行
- **THEN** 两条路径 MUST 使用一致的 batch key、sequence padding、future slot 选择和 model output 适配语义
- **AND** validation metrics MUST 不依赖独立复制的 task forward 分支

### Requirement: Distillation 算法层不得构建运行对象
`kd_sensing.distillation` 中的算法模块 MUST 专注于张量级 loss、feature/logit 对齐、confidence、ranking 和 schedule 计算。算法模块 MUST 不负责构建模型、解析 checkpoint registry、读取 dataset、准备 batch 输入或选择 device；这些运行时职责 MUST 位于 `kd_sensing.engine` 或更低层 runtime 模块。

#### Scenario: G2D 算法模块保持纯算法职责
- **WHEN** 开发者查看 `kd_sensing.distillation.g2d`
- **THEN** 该模块 MUST 不导入 model builder、checkpoint loader、artifact registry、dataset builder 或 batch preparation 模块
- **AND** G2D teacher ensemble 构建和 checkpoint 解析 MUST 位于 engine runtime 或训练扩展模块

#### Scenario: Distillation 工具轻量导入
- **WHEN** 开发者导入 G2D 或 SMP 的张量级工具函数
- **THEN** 导入 MUST 不触发默认组件注册、模型构建、checkpoint 解析或数据集读取

### Requirement: 诊断可视化不得集中在 core 聚合实现
诊断可视化实现 MUST 将配置解析、数据集准备、样本选择、统计汇总、渲染和文件写出放在对应子模块中。`diagnostics.visualization.core` MAY 保留为公开入口编排或兼容 facade，但 MUST 不再作为这些职责的主要实现聚合文件。

#### Scenario: 修改样本选择只触碰 sampling
- **WHEN** 开发者调整按 `seq_index`、label 或随机种子选择样本的策略
- **THEN** 主要变更 MUST 位于 `diagnostics.visualization.sampling` 和相关测试
- **AND** 不需要修改 render、stats、datasets 或 writers 的主要实现

#### Scenario: 修改渲染只触碰 render
- **WHEN** 开发者调整单样本 PNG 或 processed asset 的渲染布局
- **THEN** 主要变更 MUST 位于 `diagnostics.visualization.render` 和相关测试
- **AND** 不需要修改 dataset 构建、sample selection 或 metadata 写出实现

#### Scenario: core 仅承担入口编排
- **WHEN** 开发者查看 `diagnostics.visualization.core`
- **THEN** 该模块 MUST 主要负责公开入口编排、兼容导出或薄协调
- **AND** 具体配置、数据集、采样、统计、渲染和写出逻辑 MUST 能在对应子模块中找到主要实现

### Requirement: 架构增长回归检查
项目 MUST 提供快速架构回归检查，用于发现训练方法逻辑重新堆入 `trainer.py`、诊断可视化重新堆入 `core.py`、或内部代码重新依赖二级兼容聚合层的问题。该检查 MUST 可在不启动真实训练的情况下运行，并 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 检查训练主循环扩张
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证新增训练方法主要通过扩展模块接入
- **AND** 测试 MUST 防止 `trainer.py` 新增 G2D、CRAF、MARF 等方法特有的大段私有 helper

#### Scenario: 检查诊断 core 聚合回退
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证诊断可视化主要实现位于 config、datasets、sampling、stats、render 和 writers 子模块
- **AND** 测试 MUST 防止 `diagnostics.visualization.core` 再次成为主要实现聚合文件

#### Scenario: 快速检查命令可运行
- **WHEN** 开发者执行项目记录的快速架构检查命令
- **THEN** 命令 MUST 在不读取真实数据集、不加载 checkpoint、不启动训练的情况下完成
- **AND** 命令 MUST 能在全量 pytest 前暴露架构边界回归
