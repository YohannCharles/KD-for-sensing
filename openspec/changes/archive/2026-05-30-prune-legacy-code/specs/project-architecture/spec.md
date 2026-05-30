## MODIFIED Requirements

### Requirement: 包级导入不得牵出重依赖
项目 MUST 保持包级公共 API 兼容，同时避免 `__init__.py` eager import 触发重依赖运行模块。导入某个具体子模块时，系统 MUST 不因为父包初始化而额外导入训练器、dataset、诊断渲染或大型第三方依赖。已退役的 G2D、CRAF、MARF 和 Multimodal-NF 子模块 MUST 不再作为轻量导入 smoke 的保留对象。

#### Scenario: 导入 engine 轻量子模块
- **WHEN** 开发者执行 `import kd_sensing.engine.model_output`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.engine._builders_impl`
- **AND** 系统 MUST 不导入 `kd_sensing.data.transform_ops._legacy`
- **AND** 系统 MUST 不导入 `pandas` 或 `scipy`

#### Scenario: 导入 diagnostics 轻量子模块
- **WHEN** 开发者导入当前保留的 diagnostics 轻量 helper
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `kd_sensing.diagnostics.visualization.core`
- **AND** 系统 MUST 不导入 `matplotlib`
- **AND** 系统 MUST 不要求 `kd_sensing.diagnostics.g2d_diagnostics` 存在

#### Scenario: 导入 distillation 工具子模块
- **WHEN** 开发者导入当前保留的 distillation 工具子模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不因为 `kd_sensing.distillation.__init__` 导入 distiller registry、engine builder 或 dataset 转换模块
- **AND** 系统 MUST 不要求 `kd_sensing.distillation.g2d_smp` 存在

#### Scenario: 旧包级公共符号仍可访问
- **WHEN** 现有代码执行 `from kd_sensing.engine import train` 或 `from kd_sensing.diagnostics import export_viewer_manifest`
- **THEN** 导入 MUST 继续成功
- **AND** 对应重依赖模块 MUST 仅在访问该公共符号时按需加载

### Requirement: 训练方法扩展点边界
训练引擎 MUST 将后续仍被 OpenSpec 批准的方法所需的 teacher runtime、额外 loss、梯度后处理和 epoch diagnostics 接入点保持在明确模块边界内。`kd_sensing.engine.trainer` MUST 保持训练生命周期编排职责，不得作为方法特有 loss、teacher ensemble、counterfactual 或 subset-training 逻辑的主要实现位置。已退役的 G2D、CRAF 和 MARF 扩展模块 MUST 从 active code path 中删除。

#### Scenario: 新增训练方法不扩写主循环
- **WHEN** 开发者新增一个需要额外 loss 或 diagnostics 的训练方法
- **THEN** 主要实现 MUST 位于方法扩展模块及其测试中
- **AND** `kd_sensing.engine.trainer` 中的 epoch/batch 主循环 MUST 仅通过通用扩展点调用该方法

#### Scenario: 退役方法扩展模块删除
- **WHEN** 开发者查看训练期接入逻辑
- **THEN** 系统 MUST 不再保留 G2D、CRAF 或 MARF 的 teacher runtime、extra loss、subset/counterfactual forward 和 scalar diagnostics 作为 active 方法模块
- **AND** `trainer.py` MUST 不包含这些退役方法的大段私有 helper 实现

### Requirement: 共享任务 forward runtime
训练、验证、诊断预测和当前保留的 teacher runtime MUST 复用同一组任务 forward helper 来完成 batch 标准化、输入准备、model forward、输出适配和 future slot 选择。新增或修改模态输入准备、task forward 参数或强制模态 mask 行为时，变更 MUST 不要求在 trainer、validator 和 viewer prediction 中重复修改分支逻辑。已退役 G2D teacher runtime 不再属于复用对象。

#### Scenario: 修改 fusion 输入准备只改 runtime helper
- **WHEN** 开发者调整 fusion task 的 `modalities` 输入准备或 force mask 透传逻辑
- **THEN** 主要变更 MUST 限定在共享 forward runtime 模块和测试
- **AND** 不需要分别修改 trainer、validator 和 viewer prediction 的 task 分支

#### Scenario: 验证路径复用训练输入契约
- **WHEN** 训练和验证使用同一个 fusion 配置运行
- **THEN** 两条路径 MUST 使用一致的 batch key、sequence padding、future slot 选择和 model output 适配语义
- **AND** validation metrics MUST 不依赖独立复制的 task forward 分支

### Requirement: Distillation 算法层不得构建运行对象
`kd_sensing.distillation` 中的算法模块 MUST 专注于张量级 loss、feature/logit 对齐和 schedule 计算。算法模块 MUST 不负责构建模型、解析 checkpoint registry、读取 dataset、准备 batch 输入或选择 device；这些运行时职责 MUST 位于 `kd_sensing.engine` 或更低层 runtime 模块。已退役的 G2D distillation 算法模块和 SMP 工具 MUST 从支持面删除。

#### Scenario: 保留 distillation 算法保持纯算法职责
- **WHEN** 开发者查看当前保留的 `kd_sensing.distillation` 算法模块
- **THEN** 这些模块 MUST 不导入 model builder、checkpoint loader、artifact registry、dataset builder 或 batch preparation 模块
- **AND** teacher checkpoint 解析 MUST 位于 engine runtime 或训练配置构建模块

#### Scenario: Distillation 工具轻量导入
- **WHEN** 开发者导入当前保留的张量级 distillation 工具函数
- **THEN** 导入 MUST 不触发默认组件注册、模型构建、checkpoint 解析或数据集读取
- **AND** 系统 MUST 不要求 G2D 或 SMP 工具函数可导入

### Requirement: 架构增长回归检查
项目 MUST 提供快速架构回归检查，用于发现训练方法逻辑重新堆入 `trainer.py`、诊断可视化重新堆入 `core.py`、或内部代码重新依赖二级兼容聚合层的问题。该检查 MUST 可在不启动真实训练的情况下运行，并 MUST 使用 `kd_mm_beam` 环境。检查 MUST 同时防止已退役的 G2D、CRAF、MARF 和 Multimodal-NF 模块重新进入 active code path。

#### Scenario: 检查训练主循环扩张
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证新增训练方法主要通过扩展模块接入
- **AND** 测试 MUST 防止 `trainer.py` 新增退役 G2D、CRAF、MARF 等方法特有的大段私有 helper

#### Scenario: 检查诊断 core 聚合回退
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证诊断可视化主要实现位于 config、datasets、sampling、stats、render 和 writers 子模块
- **AND** 测试 MUST 防止 `diagnostics.visualization.core` 再次成为主要实现聚合文件

#### Scenario: 检查退役模块残留
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 active import、registry 和配置推荐面不再引用 G2D、CRAF、MARF 或 Multimodal-NF
- **AND** 测试 MUST 不要求这些退役模块可导入

#### Scenario: 快速检查命令可运行
- **WHEN** 开发者执行项目记录的快速架构检查命令
- **THEN** 命令 MUST 在不读取真实数据集、不加载 checkpoint、不启动训练的情况下完成
- **AND** 命令 MUST 能在全量 pytest 前暴露架构边界回归
