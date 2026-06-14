# project-architecture Specification

## Purpose
Define the package-level architecture, lightweight import boundaries, responsibility splits for training/data/diagnostics modules, and the separation between source-controlled inputs and local runtime artifacts.
## Requirements
### Requirement: 可导入包结构
项目 MUST 提供 `src/kd_sensing/` Python 包，并将数据、模型、loss、训练引擎、评估、预处理、诊断和通用工具放入职责清晰的子模块。包内模块 MUST 使用包内绝对导入或明确相对导入，不得依赖仓库根目录脚本名作为运行时导入条件。项目 MUST 不再要求或暴露 `kd_sensing.distillation` 子包。项目 MUST 不再要求或暴露 `kd_sensing.distillation` 子包。

#### Scenario: 从项目根目录导入包
- **WHEN** 开发者在项目根目录安装或设置本地包路径后执行 `import kd_sensing`
- **THEN** 导入 MUST 成功，并且不触发数据集读取、模型权重加载或训练逻辑

#### Scenario: 导入核心子模块
- **WHEN** 开发者导入 `kd_sensing.models`、`kd_sensing.data`、`kd_sensing.engine`、`kd_sensing.preprocessing` 和当前保留的 loss/evaluation 子模块
- **THEN** 每个子模块 MUST 成功导入，并暴露对应领域的公共构建入口或注册入口
- **AND** 系统 MUST 不要求 `kd_sensing.distillation` 存在

### Requirement: 模块边界清晰
项目 MUST 按职责拆分当前根目录代码：模型定义进入 `models/`，数据集与样本解析进入 `data/`，loss 进入 `losses/`、objective 或 engine 方法模块，训练/验证/测试循环进入 `engine/`，雷达和 CSV 预处理进入 `preprocessing/`，指标与 checkpoint 等通用逻辑进入 `utils/` 或 `evaluation/`。

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
- **WHEN** 训练或评估构建 dataset、model、loss、metric 或 preprocessor
- **THEN** 系统 MUST 显式导入默认组件以完成注册
- **AND** 该默认组件导入边界 MUST 不影响轻量配置加载路径

### Requirement: 横切 builder 职责拆分
训练引擎 MUST 将配置到运行对象的构建逻辑按职责拆分。dataset/dataloader 构建、启用模态推导、cache policy、归一化 artifact、run metadata、optimizer/scheduler/device 构建 MUST 有明确模块边界。已拆分的 builder 功能 MUST 通过对应窄模块使用，项目 MUST 不再保留 `the builder facade module` 兼容 facade。

#### Scenario: 修改 cache policy 不触碰 optimizer 构建
- **WHEN** 开发者调整 image 或 LiDAR cache policy 解析逻辑
- **THEN** 变更 MUST 限定在 cache policy 相关模块及其测试
- **AND** 不需要修改 optimizer、scheduler 或 device 构建逻辑

#### Scenario: 修改 normalization artifact 不触碰 dataset 模态解析
- **WHEN** 开发者调整 GPS、LiDAR 或 mmWave 归一化 artifact 的保存和加载格式
- **THEN** 变更 MUST 限定在归一化 artifact 相关模块及其测试
- **AND** 不需要修改启用模态推导逻辑

#### Scenario: 旧 builders import 被拒绝
- **WHEN** 现有代码尝试从 `the builder facade module` 导入公开构建函数
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `engine.data_factory`、`engine.optim`、`engine.cache_policy`、`engine.normalization_artifacts` 或其它对应窄模块

### Requirement: 模态数据转换职责拆分
数据转换模块 MUST 按 image、radar、lidar、gps、mmwave 和通用 IO/cache/normalization 职责组织。新增或修改某个模态的数据读取、特征构造或 cache key 时，变更 MUST 不要求编辑其它模态的转换实现。项目 MUST 不再保留 `the transform facade module` 或 `the transform aggregate module` 作为兼容聚合入口。

#### Scenario: 修改 GPS 特征不触碰 LiDAR 转换
- **WHEN** 开发者修改 GPS feature sequence 构造
- **THEN** 变更 MUST 限定在 GPS 转换相关模块和测试
- **AND** 不需要修改 LiDAR BEV、image 或 mmWave feature 转换实现

#### Scenario: 旧 transforms import 被拒绝
- **WHEN** 现有代码从 `the transform facade module` 或 `the transform aggregate module` 导入转换函数或 scaler
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `kd_sensing.data.transform_ops.<modality>` 或通用 transform 子模块

### Requirement: 诊断可视化内部模块化
诊断可视化入口 MUST 收敛为包内 viewer manifest 导出和 JEPA visual analysis。项目 MUST 不再维护旧静态 modality visualization PNG workflow、`kd_sensing.diagnostics.visualization` 内部渲染模块或仓库级 Gradio viewer support。`kd-sensing-visualize-modalities` MAY 作为兼容薄 alias 保留，但 MUST 委托 `kd-sensing-export-viewer-manifest`，不得恢复独立 parser、旧 PNG 总览图或 `tools/visualization/` support 依赖。

#### Scenario: viewer manifest 导出入口兼容
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: modality visualization 兼容 alias 不恢复 PNG workflow
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest
- **AND** 该入口 MUST 不导入 `kd_sensing.diagnostics.visualization.core` 或仓库级 `tools/visualization` helper

#### Scenario: JEPA visual analysis 作为论文图出口
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`
- **THEN** 命令 MUST 正常退出
- **AND** 该入口 MUST 使用 `kd_sensing.diagnostics.jepa_visual_analysis` 生成本地分析 manifest、图表、表格和 report

### Requirement: 源码与实验产物边界
项目 MUST 明确源码、配置、文档、OpenSpec artifacts 与本地数据、训练日志、缓存和输出产物的边界。本地运行产物 MUST 保持在 `.gitignore` 覆盖范围内，文档 MUST 指明哪些目录是可复现输入、哪些目录是可删除生成物。用户明确要求退役并清理某条失败实验路线或数据集工作流时，系统 MAY 删除匹配的本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和训练诊断产物，但 MUST 先生成可审计清单并限制在未纳入源码且属于目标路线的路径内。

#### Scenario: 本地产物不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令
- **THEN** 生成的 logs、outputs、cache、checkpoint 和 Python bytecode 产物 MUST 位于忽略规则覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

#### Scenario: 文档说明产物边界
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `dataset/`、`All_models/`、`outputs/`、`logs/` 和 cache 目录的角色
- **AND** 文档 MUST 指明哪些目录通常不应纳入源码变更
- **AND** 文档 MUST 明确用户未要求清理时，源码删除不应自动清理历史 `outputs/`

#### Scenario: 清理旧失败实验产物
- **WHEN** 用户明确要求删除已退役失败路线的输出日志和实验结果
- **THEN** 清理流程 MUST 先写出 machine-readable manifest，记录每个候选路径、匹配原因、产物类型和大小
- **AND** 清理流程 MUST NOT 删除 `dataset/`、`All_models/` 已跟踪权重、OpenSpec artifacts、源码文件或未匹配失败路线的活跃实验产物

#### Scenario: 清理退役数据集工作流
- **WHEN** 用户明确要求删除 Raymobtime s008 代码和数据集
- **THEN** 清理流程 MUST 先写出 machine-readable manifest，记录每个 Raymobtime s008 候选数据、cache、日志、checkpoint、诊断和输出路径
- **AND** 清理流程 MUST 只删除 manifest 中属于 Raymobtime s008 的允许路径
- **AND** 清理流程 MUST NOT 删除其它数据集、外部未知 data_root、`All_models/` 已跟踪权重、OpenSpec artifacts 或非 Raymobtime 活跃实验产物

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

#### Scenario: distillation 子包不再作为 smoke 对象
- **WHEN** 开发者运行轻量导入 smoke
- **THEN** 检查 MUST 不导入 `kd_sensing.distillation`
- **AND** 系统 MUST 不要求 `kd_sensing.distillation.g2d_smp` 存在

#### Scenario: 旧包级公共符号仍可访问
- **WHEN** 现有代码执行 `from kd_sensing.engine import train` 或 `from kd_sensing.diagnostics import export_viewer_manifest`
- **THEN** 导入 MUST 继续成功
- **AND** 对应重依赖模块 MUST 仅在访问该公共符号时按需加载

### Requirement: builder 实现不得集中在私有聚合模块
训练引擎 MUST 将 builder 实现放在对应职责模块中。`the private builder aggregate` 和 `the builder facade module` MUST 不再作为实现聚合或兼容转发层存在；新实现和测试 MUST 以 `cache_policy`、`modality_resolution`、`data_factory`、`normalization_artifacts`、`run_metadata` 和 `optim` 等窄模块为主。

#### Scenario: cache policy 实现在 cache 模块
- **WHEN** 开发者查看或修改 cache policy 解析逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.cache_policy`
- **AND** 不需要编辑 optimizer、run metadata 或 dataset 构建实现

#### Scenario: optimizer 和 device 构建实现在 optim 模块
- **WHEN** 开发者查看或修改 optimizer、scheduler 或 device 构建逻辑
- **THEN** 主要实现 MUST 位于 `kd_sensing.engine.optim`
- **AND** 不需要编辑 dataset/dataloader 构建实现

#### Scenario: builders facade 已删除
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 `the builder facade module` 和 `the private builder aggregate` 不再被内部代码引用
- **AND** 测试 MUST 验证构建流程仍能通过窄模块完成

### Requirement: 模态转换实现不得集中在 legacy 聚合模块
数据转换模块 MUST 将仍在使用的 image RGB、GPS、LiDAR、mmWave、radar、IO、cache 和 normalization 实现放入对应模块。`the transform aggregate module` MUST 不再存在或不再作为运行时入口导出任何符号。

#### Scenario: 修改 image 实现不触碰 LiDAR 实现
- **WHEN** 开发者修改 RGB image 加载或标准化逻辑
- **THEN** 主要变更 MUST 限定在 image 转换相关模块和测试
- **AND** 不需要编辑 LiDAR、GPS、mmWave 或 radar 转换实现

#### Scenario: 修改 GPS scaler 不触碰 image 实现
- **WHEN** 开发者修改 GPS feature 或 scaler 加载保存逻辑
- **THEN** 主要变更 MUST 限定在 GPS 或通用 normalization 模块
- **AND** 不需要编辑 image、LiDAR BEV 或 radar map 转换实现

#### Scenario: legacy 聚合模块引用被拒绝
- **WHEN** 开发者运行内部引用扫描
- **THEN** 扫描 MUST 拒绝 `the transform aggregate module`
- **AND** 扫描 MUST 指向对应的窄 transform 模块作为迁移路径

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的兼容 console script MUST 是薄 alias，不得复制长期维护的 parser 或主实现。项目 MUST 不再要求安装 `kd-sensing-raymobtime-analysis`、GPS window baseline 或仓库级 Gradio viewer support 入口。BeamBench 相关 console scripts MAY 保持当前声明。

#### Scenario: 可视化 manifest 导出入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: 可视化兼容入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest
- **AND** 该入口 MUST 委托 manifest 导出 CLI，不得复制独立 parser、旧静态 PNG 主流程或仓库级 Gradio viewer support

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-visualize-modalities`、`kd-sensing-export-viewer-manifest` 和 `kd-sensing-jepa-visual-analysis`
- **AND** 安装生成的 entry points MUST 不要求包含 `kd-sensing-raymobtime-analysis`、`kd-sensing-gps-window-baseline` 或仓库级 Gradio viewer support 入口

### Requirement: 内置权重与本地产物边界明确
项目 MUST 明确区分内置复现权重和本地生成 checkpoint。已跟踪的 `All_models` 权重如果继续保留，MUST 被文档标记为内置复现输入；新训练、评估或诊断产生的 checkpoint 和缓存 MUST 继续被忽略。

#### Scenario: README 说明 All_models 策略
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `All_models` 中已跟踪权重的用途、加载路径和是否属于源码仓库的可复现输入
- **AND** 文档 MUST 说明新生成的 `.pth` checkpoint 不应进入源码变更

#### Scenario: 新生成 checkpoint 不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令并生成 `.pth`、cache 或输出文件
- **THEN** 这些文件 MUST 位于 `.gitignore` 覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

### Requirement: 项目健康检查可分层运行
项目 MUST 提供或记录一组快速健康检查，用于在不启动真实训练的情况下验证导入边界、CLI 入口和当前保留的核心诊断逻辑。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。健康检查 MUST 不再要求 Phase 1.5 或互补分析测试存在。

#### Scenario: 轻量导入 smoke
- **WHEN** 开发者运行项目健康检查中的轻量导入 smoke
- **THEN** 检查 MUST 验证配置、路径、模态契约、engine 轻量子模块和 diagnostics 轻量子模块可导入
- **AND** 检查 MUST 验证这些导入不触发指定重依赖模块

#### Scenario: 快速回归命令覆盖当前红点
- **WHEN** 开发者运行项目健康检查中的快速回归命令
- **THEN** 检查 MUST 覆盖架构导入边界、console script help 和当前仍保留的核心诊断逻辑
- **AND** 命令 MUST 能在全量 pytest 之前快速暴露项目结构回归

#### Scenario: 全量测试仍作为最终验收
- **WHEN** 变更实现完成
- **THEN** 开发者 MUST 使用 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 全量测试 MUST 通过

### Requirement: 内部代码不得新增二级兼容聚合层依赖
项目 MUST 区分公开兼容 facade 和私有二级兼容聚合层。新内部代码 MUST 不再引用 `kd_sensing.engine._builders_impl` 或 `kd_sensing.data.transform_ops._legacy`；需要 builder 或 transform 功能时 MUST 使用窄模块或公开 facade。二级兼容聚合层若继续存在，MUST 只服务历史私有 import 过渡。

#### Scenario: 内部代码使用窄 builder 模块
- **WHEN** 开发者新增或修改训练、评估、诊断脚本
- **THEN** 代码 MUST 优先从 `engine.data_factory`、`engine.optim`、`engine.run_metadata`、`engine.cache_policy` 或其它窄模块导入
- **AND** 不得新增对 `kd_sensing.engine._builders_impl` 的依赖

#### Scenario: 内部代码使用模态 transform 模块
- **WHEN** 开发者新增或修改数据集、预处理或诊断读取逻辑
- **THEN** 代码 MUST 优先从 `kd_sensing.data.transform_ops.<modality>` 或通用 transform 子模块导入
- **AND** 不得新增对 `kd_sensing.data.transform_ops._legacy` 的依赖

### Requirement: 重复 CLI 脚本不得作为推荐入口
当包内 CLI 与 `tools/` 脚本提供同一工作流时，项目 MUST 以包内 CLI 或 `python -m kd_sensing.cli.<name>` 作为推荐入口。已被包内 CLI 覆盖的重复 fallback wrapper MUST 删除；仍保留的研究脚本或薄 alias MUST 有明确生命周期分类。

#### Scenario: viewer manifest 推荐包内 CLI
- **WHEN** 文档或 orchestration 脚本需要导出 viewer manifest
- **THEN** 推荐命令 MUST 使用 `kd-sensing-export-viewer-manifest` 或包内 CLI 模块
- **AND** 项目 MUST 不再保留 `tools/visualization/export_viewer_manifest.py` fallback wrapper

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

### Requirement: Distillation 子包已移除
项目 MUST 不再要求或暴露 `kd_sensing.distillation` 子包。监督 soft target、beam smoothing、adapter/prototype loss 和其它当前训练方法 MUST 通过 `losses`、objective 或 training extension 表达。

#### Scenario: distillation 子包导入不可作为要求
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 不要求 `kd_sensing.distillation` 可导入
- **AND** 默认组件导入 MUST 不导入 distiller registry

### Requirement: Viewer manifest 实现不得集中在聚合模块
Viewer manifest 实现 MUST 将配置解析、数据集准备、样本选择、统计汇总、schema/cache/path/merge/writer 放在对应 `viewer_manifest_*` 子模块中。`kd_sensing.diagnostics.viewer_manifest` MUST 保留为公开入口编排，不得回流承载这些职责的主要实现。

#### Scenario: 修改样本选择只触碰 sampling
- **WHEN** 开发者调整按 `seq_index`、label 或随机种子选择样本的策略
- **THEN** 主要变更 MUST 位于 `kd_sensing.diagnostics.viewer_manifest_sampling` 和相关测试
- **AND** 不需要修改 stats、datasets、schema、merge 或 writer 的主要实现

#### Scenario: 修改 asset 写出只触碰 writer
- **WHEN** 开发者调整 raw/processed asset 或 manifest record 写出
- **THEN** 主要变更 MUST 位于 `kd_sensing.diagnostics.viewer_manifest_writer` 和相关测试
- **AND** 不需要修改 dataset 构建、sample selection 或 prediction merge 实现

#### Scenario: viewer_manifest 仅承担入口编排
- **WHEN** 开发者查看 `kd_sensing.diagnostics.viewer_manifest`
- **THEN** 该模块 MUST 主要负责公开入口编排、兼容导出或薄协调
- **AND** 具体配置、数据集、采样、统计、schema/cache/path/merge/writer 逻辑 MUST 能在对应子模块中找到主要实现

### Requirement: 架构增长回归检查
项目 MUST 提供快速架构回归检查，用于发现训练方法逻辑重新堆入 `trainer.py`、viewer manifest 逻辑重新堆入公开编排模块、或内部代码重新依赖二级兼容聚合层的问题。该检查 MUST 可在不启动真实训练的情况下运行，并 MUST 使用 `kd_mm_beam` 环境。检查 MUST 同时防止已退役的 G2D、CRAF、MARF、Multimodal-NF、旧静态 visualization、GPS window 和 DeepVerse/DT31 模块重新进入 active code path。

#### Scenario: 检查训练主循环扩张
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证新增训练方法主要通过扩展模块接入
- **AND** 测试 MUST 防止 `trainer.py` 新增退役 G2D、CRAF、MARF 等方法特有的大段私有 helper

#### Scenario: 检查 viewer manifest 聚合回退
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 viewer manifest 主要实现位于 `viewer_manifest_*` 子模块
- **AND** 测试 MUST 防止旧 `kd_sensing.diagnostics.visualization` 包或仓库级 `tools/visualization` viewer support 回流

#### Scenario: 检查退役模块残留
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 active import、registry 和配置推荐面不再引用 G2D、CRAF、MARF、Multimodal-NF、GPS window、DeepVerse/DT31 或旧静态 visualization
- **AND** 测试 MUST 不要求这些退役模块可导入

#### Scenario: 快速检查命令可运行
- **WHEN** 开发者执行项目记录的快速架构检查命令
- **THEN** 命令 MUST 在不读取真实数据集、不加载 checkpoint、不启动训练的情况下完成
- **AND** 命令 MUST 能在全量 pytest 前暴露架构边界回归

### Requirement: 兼容冗余入口已删除
项目 MUST 删除已经迁移到 canonical 模块的兼容入口。源码、测试、文档和推荐命令 MUST 不再依赖 `the builder facade module`、`the transform facade module`、`the transform aggregate module`、场景专用 dataset 兼容模块或复制旧实现的可视化脚本入口。明确保留的 console-script 兼容入口 MUST 作为薄 alias 存在，并 MUST 指向当前包内主实现。

#### Scenario: 兼容 facade 不再作为公开入口
- **WHEN** 开发者在源码、测试、README 或扩展指南中搜索已删除的兼容 facade
- **THEN** 不得出现 `the builder facade module`、`the transform facade module` 或 `the transform aggregate module` 的运行时引用
- **AND** 对应功能 MUST 通过职责明确的窄模块导入

#### Scenario: 旧入口引用检查
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 拒绝新增 `scene-specific dataset class alias`、`the scene-9 dataset-type spelling`、legacy fusion 配置路径或兼容 facade 引用
- **AND** 检查 MUST 在不读取真实数据和不加载 checkpoint 的情况下完成

#### Scenario: 保留的可视化兼容入口是薄 alias
- **WHEN** 项目保留 `kd-sensing-visualize-modalities` console script
- **THEN** 该入口 MUST 调用 `kd_sensing.cli.export_viewer_manifest` 或等价当前包内主实现
- **AND** 该入口 MUST 不承载独立业务逻辑、重复 parser 或旧静态 PNG 总览图主流程

### Requirement: 训练编排层保持窄职责
训练主循环 MUST 只协调 epoch、checkpoint、optimizer、scheduler、extension hook、validation 调用和运行产物写出。objective metric alias、available metric 计算、TensorBoard objective 字段、validation forward/loss/collect 和 canonical overlay 生成 MUST 位于对应窄模块。

#### Scenario: 新增 objective 不修改 trainer 主循环
- **WHEN** 开发者新增一个 prediction objective 并完成 objective metadata、loss 和 metrics 实现
- **THEN** 不得要求修改 trainer 主循环中的 early stopping alias 表、history 字段表或 TensorBoard objective 字段表
- **AND** trainer MUST 通过 objective metadata 自动记录该 objective 的 primary metric 和日志字段

#### Scenario: 修改 validation 指标不修改 trainer 主循环
- **WHEN** 开发者修复 validation pass 中某个 objective 指标的聚合方式
- **THEN** 变更 MUST 限定在 evaluation pass、objective metrics 或 evaluation metrics 模块
- **AND** 不需要编辑 trainer 主循环

### Requirement: 启用模态解析唯一来源
训练、验证、评估、诊断和 dataset 构建路径 MUST 使用 `engine.modality_resolution` 或其公开 helper 解析启用模态。入口层不得新增 `_uses_gps`、`_uses_lidar`、`_uses_mmwave` 等重复配置推导 helper。

#### Scenario: evaluator 复用 modality resolution
- **WHEN** 评估入口需要判断当前配置是否启用 LiDAR 或 mmWave
- **THEN** 入口 MUST 调用统一模态解析 helper
- **AND** 不得在 evaluator 中维护独立的配置字段判断逻辑

#### Scenario: fusion teacher/student 模态冲突错误一致
- **WHEN** fusion 配置中 teacher 和 student modalities 不一致且未声明支持跨模态蒸馏
- **THEN** 训练和评估路径 MUST 抛出一致的错误信息
- **AND** 错误 MUST 来自统一模态解析逻辑

### Requirement: models 包级延迟导出
`kd_sensing.models` MUST 保持公开符号兼容，同时通过延迟导入暴露重依赖模型类。导入 `kd_sensing.models` 本身 MUST 不 eager import fusion、GPS、LiDAR、mmWave、image encoder 或其它模型实现模块。

#### Scenario: 轻量导入 models 包
- **WHEN** 开发者执行 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入各模型实现模块

#### Scenario: 按需访问公开模型符号
- **WHEN** 开发者执行 `from kd_sensing.models import FusionTeacherModalityNet`
- **THEN** 系统 MUST 按需导入对应实现模块并返回该公开符号
- **AND** `__all__` 中的既有公开模型符号 MUST 继续可访问

#### Scenario: removed alias 错误保持兼容
- **WHEN** 现有代码访问已移除的模型别名
- **THEN** `kd_sensing.models` MUST 继续抛出清晰 `AttributeError`
- **AND** 错误信息 MUST 指向替代公开符号

### Requirement: 训练运行时编排职责拆分
训练引擎 MUST 将训练运行时状态、单 batch step、epoch metrics/history、checkpoint/sidecar、TensorBoard 和最终 artifact 写出拆到职责明确的窄模块或 helper。`kd_sensing.engine.trainer.train` MAY 保留为公开入口和顶层生命周期编排器，但 MUST 不继续直接承载这些细节的主要实现。

#### Scenario: batch step 逻辑位于窄模块
- **WHEN** 开发者查看训练中单 batch 的 prepare、forward、loss、backward 和 optimizer step 编排
- **THEN** 主要实现 MUST 位于 batch step runner 或等价窄模块
- **AND** `trainer.py` MUST 只负责调用该 runner 并消费其返回的 loss、diagnostics 和状态更新

#### Scenario: checkpoint 写出位于 checkpoint manager
- **WHEN** 开发者调整 `best.pth`、`best_top1.pth`、`last.pth`、sidecar 或 checkpoint registry archive 的写出逻辑
- **THEN** 主要变更 MUST 限定在 checkpoint manager 或等价窄模块
- **AND** 不需要编辑训练 batch 主循环

#### Scenario: 训练 artifact 写出位于 artifact writer
- **WHEN** 开发者调整 `train_log.json`、`training_outputs.npz`、`final_config.yaml`、训练曲线或 debug artifact 的写出逻辑
- **THEN** 主要变更 MUST 限定在 artifact writer、history recorder 或等价窄模块
- **AND** 不需要编辑模型 forward、KD loss 或 optimizer step 逻辑

### Requirement: config/io 不承载业务规则实现
`kd_sensing.config.io` MUST 保持配置入口协调职责，负责加载实体 YAML 或 virtual config、应用命令行覆盖、调用 normalization pipeline 和调用 validation pipeline。objective 默认补全、模态推导、dataset-specific rules、迁移拒绝和 schema validation 的主要实现 MUST 位于独立 helper。

#### Scenario: Raymobtime 退役规则不写在 io 入口
- **WHEN** 开发者调整 Raymobtime s008 退役配置、旧 dataset 名称或旧 preprocessor 名称的拒绝规则
- **THEN** 主要实现 MUST 位于 migration guard、config validation helper 或 registry 拒绝 helper
- **AND** `config/io.py` MUST 只调用该 helper，不得恢复 Raymobtime dataset/preprocessor 运行路径

#### Scenario: removed image motion guard 不写在 io 入口
- **WHEN** 开发者调整已删除 image motion profile、cache 或 encoder 的拒绝逻辑
- **THEN** 主要实现 MUST 位于 migration guard 或 image profile validation helper
- **AND** `config/io.py` MUST 不直接维护该迁移规则的完整实现

#### Scenario: objective 默认值不写在 io 入口
- **WHEN** 开发者新增或调整 prediction objective 的默认 early stopping metric、loss weights 或 required target/head
- **THEN** 主要实现 MUST 位于 objective metadata、normalization helper 或 validation helper
- **AND** `config/io.py` MUST 不维护 objective 专属分支表

### Requirement: 项目表面积回归检查
项目 MUST 提供轻量表面积回归检查，用于发现源码变更中重新引入的本地产物、重复入口、已删除兼容路径和可生成配置实体化。该检查 MUST 不读取真实数据集、不加载 checkpoint、不启动训练，并 MUST 使用 `kd_mm_beam` 环境运行。

#### Scenario: 本地产物未进入源码表面积
- **WHEN** 开发者运行表面积回归检查
- **THEN** 检查 MUST 拒绝已跟踪的 `__pycache__`、`.pyc`、`.pytest_cache`、训练输出、日志、cache 和新生成 checkpoint
- **AND** 检查 MUST 允许 `dataset/.gitkeep` 这类明确的源码占位文件

#### Scenario: 重复入口回流被拒绝
- **WHEN** 项目中新增 `scripts/` 或 `tools/` Python 入口
- **THEN** 检查 MUST 判断该入口是否复制已有 `kd_sensing.cli.*` parser/main 或 console script 工作流
- **AND** 重复入口 MUST 被拒绝，除非对应 OpenSpec requirement 明确允许该薄 alias 或研究脚本边界

#### Scenario: 表面积 inventory 可审计
- **WHEN** 开发者运行架构边界测试或专用 inventory 命令
- **THEN** 输出或测试断言 MUST 覆盖实体 YAML 数量、脚本入口数量、README/OpenSpec 待整理项和已知兼容入口 allowlist
- **AND** 新增 allowlist 项 MUST 通过 OpenSpec change 说明原因

### Requirement: 重复开发入口必须有生命周期
当包内 CLI 或 console script 已覆盖同一工作流时，项目 MUST 删除对应 `scripts/` 或 `tools/` fallback wrapper，或者在 OpenSpec 中明确其短期保留原因和删除条件。保留的研究脚本 MUST 不作为 README 推荐入口。

#### Scenario: manifest 导出 fallback wrapper 删除
- **WHEN** `kd-sensing-export-viewer-manifest` 和 `python -m kd_sensing.cli.export_viewer_manifest` 可用
- **THEN** 项目 MUST 不再要求保留 `tools/visualization/export_viewer_manifest.py` 作为 fallback wrapper
- **AND** README 和工具文档 MUST 推荐包内 CLI 或 console script

#### Scenario: 研究脚本保留边界清晰
- **WHEN** `scripts/` 或 `tools/analysis/` 中的脚本没有等价包内 CLI
- **THEN** 该脚本仅可作为研究/诊断工具保留
- **AND** 文档 MUST 不把该脚本描述为训练、评估、预处理或 manifest 导出的唯一推荐入口

### Requirement: 文档与 OpenSpec 沉积必须可整理
README、docs 和 OpenSpec MUST 按职责维护当前行为，不得长期保留只描述历史迁移过程且不定义当前需求的正文。Archived spec 中的 TBD purpose MUST 被补齐或在后续归档整理中移除。

#### Scenario: README 保持入口导向
- **WHEN** 开发者阅读 README
- **THEN** README MUST 提供安装、环境、健康检查、主要入口和数据/产物边界
- **AND** 长实验矩阵、分析流程和 viewer 操作细节 MUST 通过 docs 或 OpenSpec 链接承载

#### Scenario: specs purpose 完整
- **WHEN** 开发者运行 OpenSpec 文档健康检查
- **THEN** 检查 MUST 拒绝新增 `TBD - created by archiving` purpose
- **AND** 既有 TBD purpose MUST 在本次整理范围内被替换为当前 capability 的真实目的说明

### Requirement: Objective 元数据轻量导入边界
项目 MUST 将 prediction objective 的纯元数据契约与 torch loss/runtime 实现解耦。导入配置加载、配置 normalization 或配置 validation 路径时，系统 MUST 能读取 objective 默认 metric、metric mode、required target/head、history fields 和 TensorBoard scalar 映射，且不得因此导入 torch、模型、dataset、诊断渲染或训练主循环。

#### Scenario: 配置导入不触发 torch
- **WHEN** 开发者执行 `import kd_sensing.config`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `torch`
- **AND** 系统 MUST 不导入 dataset 类、模型实现、诊断可视化 core 或训练主循环

#### Scenario: runtime loss 仍可通过原入口使用
- **WHEN** 训练或验证代码从 `kd_sensing.engine.prediction_objectives` 导入 prediction target 和 loss helper
- **THEN** 导入 MUST 继续成功
- **AND** torch loss 计算语义 MUST 与变更前保持兼容
- **AND** objective 元数据 MUST 来自同一轻量契约，避免配置路径和 runtime 路径维护两套表

### Requirement: Viewer manifest 轻量 helper import 边界
Viewer manifest 内部模块 MUST 按职责控制 import 边界。配置解析和采样选择等轻量 helper MUST 不导入 matplotlib、PIL、dataset builder、model builder 或训练 runtime。数据集构建、统计汇总、processed asset 写出和模型预测导出等重依赖职责 MUST 留在对应运行模块或函数内部。

#### Scenario: 导入 manifest 配置 helper 不触发渲染栈
- **WHEN** 开发者执行 `import kd_sensing.diagnostics.viewer_manifest_config`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 `matplotlib`
- **AND** 系统 MUST 不导入 `PIL.Image`
- **AND** 系统 MUST 不导入 `kd_sensing.engine.data_factory`

#### Scenario: 导入采样 helper 不构建数据集
- **WHEN** 开发者导入 `kd_sensing.diagnostics.viewer_manifest_sampling`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset builder、model builder、writer 或旧 visualization core
- **AND** 该模块 MUST 只处理候选样本过滤、随机选择和 JSON 标量规范化

### Requirement: OpenSpec 文档健康检查结构化
项目 MUST 使用结构化方式检查 OpenSpec capability purpose。健康检查 MUST 只检查每个 spec 的 `## Purpose` 段落是否为空、过短或仍为归档占位文本，不得因为正文中描述被拒绝的占位文本而误判。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: purpose 检查不自引用误伤
- **WHEN** 某个 spec 正文描述健康检查会拒绝归档占位文本
- **THEN** 健康检查 MUST 不因正文出现该字符串而失败
- **AND** 检查 MUST 只根据 `## Purpose` 段落判断该 spec 是否存在文档健康问题

#### Scenario: purpose 问题报告具体 spec
- **WHEN** 某个 spec 的 `## Purpose` 段落为空、过短或仍为归档占位文本
- **THEN** 健康检查 MUST 报告该 spec 路径
- **AND** 报告 MUST 指向需要补齐的 capability purpose，而不是要求改写无关正文

### Requirement: 源码热点模块必须按职责收敛
项目 MUST 将继续增长的大文件拆分到职责明确的窄模块中。拆分后，公开入口 MAY 保留薄 facade 或兼容导出，但主要实现 MUST 位于按职责命名的模块中，不得重新形成新的私有聚合层。已退役的互补性分析模块 MUST 不再作为源码热点分层要求。

#### Scenario: 修改 manifest 过滤逻辑不触碰 asset 写出
- **WHEN** 开发者调整 viewer manifest 的 scene、split、sample limit 或低质量样本过滤逻辑
- **THEN** 主要变更 MUST 位于 viewer manifest 过滤、cache 或 IO 相关模块
- **AND** 不需要修改 processed asset 写出、prediction summary 合并或 JEPA visual analysis 图表实现

#### Scenario: 修改 CSI hardening 不触碰 tokenizer
- **WHEN** 开发者调整 CSI hardening、pilot estimation 或噪声诊断逻辑
- **THEN** 主要变更 MUST 位于 CSI estimation 或 hardening 相关模块
- **AND** 不需要修改 CSI view tokenizer、view fusion 或 encoder registry glue

### Requirement: Raymobtime s008 预处理退役边界
Raymobtime s008 预处理 workflow 已退役，不属于当前源码支持面。项目 MUST 不再暴露 Raymobtime s008 preprocessor registry、实体配置、dataset/model 实现或 focused test；旧名称只能通过 migration guard、registry 拒绝或历史说明出现。

#### Scenario: Raymobtime 预处理入口不可用
- **WHEN** 用户引用 Raymobtime s008 预处理配置、preprocessor type 或 `raymobtime_s008` dataset type
- **THEN** 配置加载或 registry lookup MUST fail fast
- **AND** 错误信息 MUST 明确 Raymobtime s008 已退役且无兼容迁移入口

#### Scenario: Raymobtime 源码不回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 当前源码 MUST 不重新引入 Raymobtime s008 dataset、preprocessor、selection model、配置或测试
- **AND** 本地 `dataset/` 或历史 archive 中存在 Raymobtime 资料 MUST 不被解释为当前支持能力

### Requirement: 入口生命周期必须可审计
项目 MUST 为 `scripts/` 和 `tools/analysis/` 中保留的入口维护生命周期分类。新增或保留入口 MUST 属于包内 CLI、薄 alias、研究诊断脚本、数据准备脚本或 shell orchestration 中的一类，并在架构检查 allowlist 或 inventory 中记录原因。仓库级 `tools/visualization/` viewer support 已退役，MUST 不再作为当前入口分类回流。

#### Scenario: 新增脚本入口需要分类
- **WHEN** 开发者新增 `scripts/` 或 `tools/analysis/` 下的 Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在生命周期 allowlist 或 inventory 中
- **AND** 如果该入口复制已有 console script 或包内 CLI 工作流，检查 MUST 拒绝该入口

#### Scenario: 重复 wrapper 不作为推荐入口
- **WHEN** 包内 CLI 或 console script 已覆盖训练、评估、预处理或 viewer manifest 导出工作流
- **THEN** README 和工具文档 MUST 推荐包内 CLI 或 console script
- **AND** 对应 `scripts/` 或 `tools/` fallback wrapper MUST 删除或被明确标注为短期薄 alias

### Requirement: 架构优化不得触碰本地数据和产物
源码、配置和入口表面积优化 MUST 不移动、删除、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或其它本地运行产物。相关检查 MUST 只验证源码控制范围内的文件和忽略规则。

#### Scenario: 实施源码拆分不清理产物
- **WHEN** 开发者实施本 change 中的源码拆分、配置瘦身或入口收敛任务
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/` 中真实文件的删除、移动或压缩操作
- **AND** 架构检查 MUST 继续只拒绝已跟踪源码表面积中的本地产物污染

#### Scenario: 数据目录策略不随本变更改变
- **WHEN** 本 change 完成并归档
- **THEN** 默认数据目录、legacy data_root 兼容规则和用户显式 data_root 行为 MUST 保持不变
- **AND** 本 change MUST 不要求用户迁移本地数据才能继续运行既有配置

### Requirement: 热点模块拆分边界
项目 MUST 为高变更频率的大型模块提供职责拆分路径。拆分后的窄模块 MUST 按 schema/constants、pure helper、reader、writer、orchestration 或 domain-specific adapter 组织，公开 facade MAY 保留兼容导出，但新内部代码 MUST 优先依赖窄模块。

#### Scenario: 新内部代码使用窄模块
- **WHEN** 开发者在训练、评估、预处理、诊断或 viewer 相关实现中新增代码
- **THEN** 新代码 MUST 优先从职责明确的窄模块 import
- **AND** 不得新增对仅用于兼容 re-export 的二级聚合模块的内部依赖

#### Scenario: 公开入口兼容
- **WHEN** 现有用户从公开 facade import 旧符号
- **THEN** 导入 MUST 继续成功，除非对应 change 明确声明 breaking change
- **AND** facade MUST 不触发比旧路径更重的 eager import

### Requirement: 热点模块 inventory 与回流防护
项目 MUST 维护热点模块拆分 inventory 或测试 allowlist，记录哪些模块仍作为兼容 facade 保留，哪些内部路径不得新增引用。架构边界测试 MUST 覆盖这些禁止回流路径。

#### Scenario: 架构测试拒绝内部 facade 回流
- **WHEN** 内部源码新增对已标记为兼容 facade 的二级聚合模块依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向推荐的窄模块路径

#### Scenario: inventory 更新
- **WHEN** 新增或拆分 scripts、tools、viewer manifest helper 或大型 domain helper
- **THEN** 项目表面积 inventory 或等价文档 MUST 记录该入口的 lifecycle 和职责
- **AND** 测试 allowlist MUST 与文档保持一致

### Requirement: 拆分后轻量导入保持
热点模块拆分 MUST 不破坏现有轻量导入边界。schema、constants、objective metadata 查询、dataset descriptor 查询和 path helper 查询 MUST 不因为拆分而导入训练循环、dataset 实例、模型、大型可视化依赖或真实数据读取逻辑。

#### Scenario: objective schema 轻量导入
- **WHEN** 开发者导入 objective metadata 的 schema/registry 子模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入训练器、dataset、模型或 matplotlib

#### Scenario: dataset runtime schema 轻量导入
- **WHEN** 开发者查询 dataset descriptor 或 runtime schema helper
- **THEN** 查询 MUST 不打开 HDF5、CSV、image、LiDAR 或 checkpoint 文件
- **AND** 查询 MUST 不导入训练循环

### Requirement: MMW 入口生命周期 inventory 必须同步
新增或保留的 MMW Python 脚本、shell orchestration 和研究支持入口 MUST 具有可审计生命周期。项目表面积 inventory 与架构边界测试 allowlist MUST 同步记录入口类别、保留原因、推荐入口关系、输出产物边界和删除或收敛条件。

#### Scenario: 新增 MMW 脚本入口需要 inventory
- **WHEN** 开发者新增 `scripts/`、`scripts/mmw/` 或 `tools/analysis/` 下的 MMW Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在项目表面积 inventory 或等价生命周期文档中
- **AND** inventory MUST 说明该入口属于包内 CLI、薄 alias、研究诊断脚本、数据准备脚本或 shell orchestration 中的哪一类
- **AND** 对应测试 allowlist MUST 与 inventory 保持一致

#### Scenario: 未登记入口导致表面积检查失败
- **WHEN** 工作区中存在未登记的 MMW Python 或 shell 入口
- **THEN** 表面积回归检查 MUST 失败
- **AND** 失败信息 MUST 列出缺失登记的相对路径
- **AND** 失败信息 MUST 指向更新 inventory、删除重复入口或改为包内 CLI 的修复路径

#### Scenario: 重复 MMW orchestration 不成为推荐入口
- **WHEN** 多个 shell orchestration 覆盖同一 MMW quick validation 工作流
- **THEN** inventory MUST 标记推荐入口和补充 profile 的关系
- **AND** README 或 docs MUST 不把重复 shell wrapper 描述为唯一 canonical 入口
- **AND** 若已有包内 CLI 覆盖同一工作流，重复 shell wrapper MUST 标记为短期薄 alias 或研究脚本

### Requirement: HiST-Beam LOSO executor 退役边界
HiST-Beam/Hist 专用 LOSO executor 已从当前支持面退役。项目 MUST 不再把 `hist_beam_loso_execution.py`、`kd-sensing-hist-beam-loso` 或 Hist 专用 run plan 描述为当前热点、当前推荐入口或待拆分 facade；通用 LOSO/few-shot split helper 如保留，只能作为 supporting 能力并由新的 current workflow 显式消费。

#### Scenario: Hist LOSO 入口不回流
- **WHEN** 开发者检查 CLI、pyproject、scripts allowlist 或包内 engine
- **THEN** 项目 MUST 不声明 Hist LOSO runner、Hist executor facade 或 Hist 默认矩阵
- **AND** 若文档提到 Hist LOSO，MUST 明确其 retired-tombstone 或 migration guard 语义

#### Scenario: 非 Hist LOSO 未来能力需要新契约
- **WHEN** 未来 workflow 需要 leave-one-scene-out、few-shot target split 或跨场景 summary
- **THEN** 新 workflow MUST 通过 current capability 明确定义 CLI、配置、输出和防泄漏边界
- **AND** 系统 MUST 不复用退役 Hist run plan 作为隐式默认

#### Scenario: 通用 helper 保留为 supporting
- **WHEN** 当前 workflow 复用通用 fold planning、target adapt/test split、few-shot sampling 或 claim guard helper
- **THEN** 文档 MUST 将这些 helper 标记为 supporting
- **AND** 文档 MUST 不把 Hist executor、Hist config 或 Hist output artifact 恢复为当前入口

### Requirement: 当前架构规格遵循 lifecycle 分类
`project-architecture` spec MUST 与 OpenSpec capability lifecycle inventory 保持一致。已经标记为 `retired-tombstone` 的能力 MUST 只作为退役边界、禁止回流、migration guard 或历史背景出现；标记为 `supporting` 的能力 MUST 不被描述为 standalone 当前推荐入口。

#### Scenario: 退役能力不作为当前热点
- **WHEN** `project-architecture` 提到 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 或旧 KD
- **THEN** 对应段落 MUST 明确其 retired/supporting 语义
- **AND** 文档 MUST 不要求恢复旧 CLI、旧配置、旧 facade 或旧 root script

#### Scenario: 支撑能力指向当前 workflow
- **WHEN** `project-architecture` 提到仍被当前 workflow 复用的支撑代码
- **THEN** 文档 MUST 指向实际 current workflow
- **AND** 文档 MUST 不把支撑代码所属的旧研究路线描述为当前入口

### Requirement: Active mainline 与 legacy KD 模块边界
项目 MUST 区分当前主线方法模块、supporting helper 和 legacy/retired 模块。当前主线包括 supervised beam prediction、Image+GPS JEPA query-pool downstream、paired baseline/control、Vision-Position baseline suite、DeepSense6G/MMW GPS+LiDAR BGAM、MMW GPS v2、CSI hardening、viewer manifest、JEPA visual analysis、GPS shortcut benchmark、soft-label supervised training 和通用训练/评估能力。HiST/Hist、GPS residual、camera residual、standalone Top8 selector、Raymobtime s008、CRAF/MARF/G2D、Multimodal-NF 和旧 KD MUST 不作为 active mainline 描述；若仍有通用 helper 被保留，MUST 标记为 supporting 或迁移边界。

#### Scenario: mainline 导入不触发 KD runtime
- **WHEN** 开发者导入当前主线的训练、评估、BGAM、JEPA downstream、CSI hardening、viewer 或 soft-label helper
- **THEN** 导入 MUST 不构建 frozen teacher runtime
- **AND** 导入 MUST 不解析 teacher checkpoint registry
- **AND** 导入 MUST 不要求 legacy KD baseline 模块可用

#### Scenario: 退役 Hist 不属于 active mainline
- **WHEN** 文档或测试列举 active mainline 方法
- **THEN** 列表 MUST 不包含 HiST-Beam/Hist 专用 CLI、engine、model、evaluation、LOSO executor 或 history-anchor Hist workflow
- **AND** 如提到 Hist 名称，MUST 明确为 retired-tombstone 或禁止回流边界

#### Scenario: 架构测试拒绝 KD 和退役路线回流
- **WHEN** 内部源码新增 active mainline 到 legacy KD runtime 聚合入口或退役路线专属模块的依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向 no-KD objective、current workflow、supporting helper 或 migration guard 作为修复路径

### Requirement: 新主线方法不得包含 distillation 配置段
新 current mainline 配置和运行时 MUST 用 `model.primary` 与 supervised/adaptation loss 表达训练。任何 `distillation.*`、`teacher_model_name`、`logits_kd`、`rkd` 或旧 `*_no_kd` 路径 MUST 在配置解析阶段失败并给出迁移建议。

#### Scenario: 当前配置无需 distillation 字段
- **WHEN** 用户加载当前推荐的 supervised/adaptation mainline 配置
- **THEN** 配置 validation MUST 不要求 `distillation.teacher_model_name`
- **AND** 最终配置 MUST 不包含 KD temperature、alpha 或 RKD 权重字段

#### Scenario: 旧 no_kd 字段被拒绝
- **WHEN** 用户加载仍包含 `distillation.type: no_kd` 的历史配置
- **THEN** 系统 MUST 拒绝该配置并提示 strong、lightweight 或 supervised 入口
- **AND** 系统 MUST 不把该 run 作为可运行 baseline

### Requirement: 当前源码热点必须收敛为薄 facade
项目 MUST 优先防止当前仍保留的大型 workflow 或公开 orchestration 入口重新聚合职责。`src/kd_sensing/data/mmw/preparation.py`、viewer manifest、BGAM workflow、evaluation pass、batch preparation 和训练主循环等当前热点 MUST 在 inventory 中记录拆分方向和预算；已退役的 Hist LOSO executor MUST 不再作为当前热点或兼容 facade 要求。

#### Scenario: Hist executor 不作为当前 facade
- **WHEN** 开发者运行架构边界测试或审阅热点 inventory
- **THEN** 检查 MUST 不要求 `hist_beam_loso_execution.py` 存在
- **AND** 文档 MUST 不把 Hist executor 作为当前待拆热点、公开入口或兼容 facade

#### Scenario: MMW preparation facade 收敛
- **WHEN** 开发者查看或修改 MMW Town10 preparation 的配置解析、zip/input 审计、sensor/channel indexing、sequence split、beam power 派生、manifest 写出、report 或 proxy geometry 逻辑
- **THEN** 主要实现 MUST 位于 `preparation_config.py`、`preparation_audit.py`、`preparation_index.py`、`preparation_splits.py`、`preparation_beam_power.py`、`preparation_writers.py`、`preparation_geometry.py` 或等价窄模块
- **AND** `data/mmw/preparation.py` MUST 只承担公开 orchestration、现有公开 helper 的兼容导出和顶层参数编排
- **AND** 架构边界测试 MUST 拒绝把上述窄职责的大段 helper 重新实现到 `data/mmw/preparation.py`

### Requirement: MMW preparation 拆分后职责不得交叉回流
MMW preparation 拆分后的窄模块 MUST 按配置、输入审计、索引、split、beam power、写出和几何/proxy 特征职责组织。新增或修改某一职责时，变更 MUST 不要求编辑其它无关职责模块，除非公开 orchestration 需要连接新的参数或结果字段。

#### Scenario: 修改 sequence split 不触碰 beam power
- **WHEN** 开发者调整 MMW sequence row 构造、group-safe split、guard band 或 leakage diagnostics
- **THEN** 主要变更 MUST 位于 `preparation_splits.py` 或等价 split 模块
- **AND** 不需要修改 channel payload 读取、DFT/codebook beam power 计算或 power vector 校验实现

#### Scenario: 修改 zip/input 审计不触碰 manifest 写出
- **WHEN** 开发者调整 MMW zip 输入校验、extract marker、source hash 或 availability audit
- **THEN** 主要变更 MUST 位于 `preparation_audit.py` 或等价审计模块
- **AND** 不需要修改 manifest CSV、split CSV、report JSON 或 artifact path 写出实现

#### Scenario: 修改 proxy geometry 不触碰 index
- **WHEN** 开发者调整 relative geometry、pose 解析、vehicle proxy feature 或 azimuth bin 逻辑
- **THEN** 主要变更 MUST 位于 `preparation_geometry.py` 或等价几何模块
- **AND** 不需要修改 sensor frame indexing、channel file indexing 或 scenario root 搜索实现

### Requirement: 热点拆分 inventory 必须覆盖优先级和禁止回流路径
项目 MUST 在表面积 inventory 或等价文档中记录热点模块拆分优先级、兼容 facade、推荐窄模块和禁止内部回流路径。架构边界测试 MUST 与 inventory 保持一致，并 MUST 对第一批 facade 执行行数上限、禁止片段和 helper 所属模块断言。

#### Scenario: inventory 记录第一批与第二梯队热点
- **WHEN** 开发者运行架构边界测试或审阅 `docs/project_surface_inventory.md`
- **THEN** inventory MUST 记录 `data/mmw/preparation.py`、viewer manifest、BGAM、trainer、dataset、run index、batch 和 evaluation pass 等当前热点的拆分方向
- **AND** inventory MUST 明确 HiST-Beam/Hist 专用 engine/model/evaluation 源码已退役，不作为当前热点清单成员
- **AND** inventory MUST 说明第二梯队热点的后续拆分方向或暂缓原因

#### Scenario: 内部代码不得从第一批 facade 回流导入 helper
- **WHEN** 内部源码新增对第一批 facade 中已迁移 helper 的 import 或调用
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应窄模块作为修复路径

### Requirement: 热点拆分必须保持公开行为兼容
热点模块拆分 MUST 保持现有公开 CLI、公开 import、manifest schema、run metadata、summary CSV/JSON、preparation artifact 命名、样本契约和默认路径策略兼容。拆分只允许改变内部模块组织，不得改变模型数值语义、数据 split 语义、beam label 语义或本地产物边界。

#### Scenario: 退役 Hist 产物只读保留
- **WHEN** 历史 HiST-Beam LOSO run metadata、summary JSON、checkpoint reuse metadata 或本地输出仍在 `outputs/` 中
- **THEN** 当前源码热点拆分 MUST 不要求这些 Hist artifact 可由当前 runner 继续生成
- **AND** cleanup/index 工具 MAY 将其作为历史或退役产物只读审计

#### Scenario: MMW preparation 产物兼容
- **WHEN** MMW preparation 拆分完成后运行 focused characterization tests
- **THEN** 现有 frame manifest、sequence split CSV、split metadata、beam power artifact、data availability report 和 report JSON 的关键字段 MUST 保持兼容
- **AND** 测试 MUST 覆盖公开 `prepare_town10_skybridge` 工作流和仍保留的公开 helper import

#### Scenario: 本地产物边界不随拆分改变
- **WHEN** 开发者实施热点拆分、运行 focused tests 或执行 CLI smoke
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或真实本地运行产物的删除、移动、压缩或重写
- **AND** 生成的临时验证产物 MUST 位于忽略规则覆盖范围内或测试临时目录中

### Requirement: 分层验证必须覆盖热点拆分
热点拆分实现完成后，项目 MUST 分层验证 OpenSpec、架构边界、focused 行为兼容和公开入口。所有项目相关 Python 验证 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 快速架构与 OpenSpec 校验
- **WHEN** 热点拆分任务完成
- **THEN** 开发者 MUST 运行 `openspec validate modularize-hotspot-modules --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 领域 focused tests 校验
- **WHEN** MMW preparation、viewer manifest、BGAM 或其它当前热点拆分完成
- **THEN** 开发者 MUST 运行对应 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`
- **AND** 若拆分触碰公开 CLI 或 viewer 入口，开发者 MUST 运行对应 help smoke 或 viewer/import smoke

#### Scenario: 全量回归作为最终验收
- **WHEN** 第一批热点拆分和架构防护全部完成
- **THEN** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 若全量测试因环境或本地数据缺失无法完成，最终说明 MUST 明确列出未运行原因和已完成的替代 focused 验证

### Requirement: 退役旧模态诊断脚本入口
项目 MUST 不再把模态失衡时期的独立模态子集和模态扰动研究脚本作为长期维护入口。通用模态 subset、mask 或 perturbation 调试能力如需保留，MUST 通过包内 CLI、配置化 evaluation pass、viewer manifest、JEPA benchmark、BGAM/CSI 当前 workflow 或明确的内部 helper 承载，并 MUST 在脚本 allowlist 和项目表面积 inventory 中体现当前边界。

#### Scenario: 脚本入口清单不包含旧诊断脚本
- **WHEN** 开发者运行架构边界测试检查 `scripts/` 与 `tools/` 入口清单
- **THEN** `scripts/eval_modality_subsets.py` 和 `scripts/eval_modality_perturbation.py` MUST 不再作为允许的长期入口存在
- **AND** 测试 MUST 继续允许当前保留的 thin CLI alias、dataset preparation、viewer manifest、MMW current workflow、BGAM、CSI hardening 和研究诊断入口

#### Scenario: 通用 subset 能力不被误删
- **WHEN** evaluation 配置启用 `evaluation.modality_subsets`
- **THEN** 系统 MUST 继续能在共享 evaluation pass 中计算配置化 subset metrics
- **AND** 该能力 MUST 不依赖被退役的独立研究脚本

### Requirement: 表面积 inventory 跟随当前主线
项目 surface inventory MUST 将当前推荐入口描述为 Image+GPS JEPA query-pool 主线、paired baseline/control、Vision-Position baseline suite、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、viewer manifest、JEPA visual analysis、GPS shortcut benchmark 和通用训练评估能力。已退役的模态失衡诊断脚本、KD virtual alias、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D 和 Multimodal-NF MUST 不作为新入口或健康检查要求出现。

#### Scenario: inventory 删除旧研究入口
- **WHEN** 开发者阅读 `docs/project_surface_inventory.md`
- **THEN** 文档 MUST 不再把旧模态子集/扰动诊断脚本或退役研究线列为长期维护 research diagnostic/current entry
- **AND** 文档 MUST 保留本地产物边界说明，不要求删除或迁移历史 `outputs/`、`logs/` 或 `dataset/`

#### Scenario: inventory 标注 supporting 能力
- **WHEN** 某个支撑代码仍被 BGAM、benchmark、metrics 或 migration guard 消费，但其 standalone workflow 已退役
- **THEN** inventory MUST 将其描述为 supporting 或支撑代码
- **AND** inventory MUST 不为该旧 workflow 新增 root config、console script 或 quickstart 命令

### Requirement: 语义化本地输出目录
项目 MUST 避免新脚本或默认配置继续向语义不清的兜底目录写入实验产物。长期保留的 shell orchestration、诊断脚本和 CLI 默认输出目录 MUST 包含实验族、数据集或能力名称；`outputs/other/` MAY 作为历史清理候选被扫描，但 MUST 不再作为新实验脚本的默认输出根。

#### Scenario: MMW modal15 默认输出目录可识别
- **WHEN** 用户直接运行 MMW modal15 shell orchestration 且未设置 `OUTPUT_ROOT`
- **THEN** 脚本 MUST 默认写入包含 `mmw_sunny_modal15` 或等价实验族名称的 `outputs/` 子目录
- **AND** 帮助文本 MUST 展示该语义化默认路径

#### Scenario: outputs other 不作为新默认值
- **WHEN** 架构边界测试扫描长期保留脚本和配置
- **THEN** 测试 MUST 拒绝新增默认输出根为 `outputs/other`
- **AND** 已存在的历史 `outputs/other/` 本地产物 MUST 只通过清理 manifest 管理

### Requirement: 清理流程不跨越源码边界
项目 MUST 将本地运行产物清理限定在 `.gitignore` 覆盖的本地产物范围内。清理工具、文档和测试 MUST 明确禁止删除源码、配置、文档、OpenSpec artifacts、已跟踪文件、`dataset/` 真实数据和 `All_models/` 历史复现权重。

#### Scenario: 清理 manifest 不含源码删除动作
- **WHEN** 用户生成清理候选 manifest
- **THEN** manifest MUST NOT 将 `src/`、`tests/`、`configs/`、`docs/` 或 `openspec/` 下的已跟踪文件列为可删除候选
- **AND** 如果这些路径被扫描到，manifest MUST 标记为 protected

#### Scenario: 文档说明本地产物边界
- **WHEN** 开发者阅读项目表面积 inventory 或 README
- **THEN** 文档 MUST 说明清理流程先生成 manifest
- **AND** 文档 MUST 说明真正删除需要用户显式确认

### Requirement: GPS+LiDAR BGAM 包内入口
项目 MUST 将 GPS+LiDAR BGAM reranker 的实现放入 `src/kd_sensing/` 包内。manifest enrich、dataset、geometry utility、model、loss、engine、evaluation、debug plot 和 CLI MUST 按现有职责边界分布在 `kd_sensing.utils`、`kd_sensing.data`、`kd_sensing.models`、`kd_sensing.losses`、`kd_sensing.engine`、`kd_sensing.evaluation` 和 `kd_sensing.cli` 中。项目 MUST NOT 新增长期维护的顶层 `train_gps_lidar_bgam.py`、`eval_gps_lidar_bgam.py`、`datasets/gps_lidar_dataset.py` 或 `models/gps_lidar_bgam.py` 旁路入口。

#### Scenario: console scripts 暴露 BGAM workflow
- **WHEN** 开发者完成 editable install 并查看 `pyproject.toml` entry points
- **THEN** 项目 MUST 暴露 GPS+LiDAR BGAM 相关 console scripts
- **AND** scripts MUST 至少覆盖 manifest enrich、训练/评估运行和独立评估
- **AND** 每个 console script MUST 委托 `kd_sensing.cli.*` 中的包内实现

#### Scenario: 包内 module CLI 可运行
- **WHEN** 用户执行 `conda run -n kd_mm_beam python -m kd_sensing.cli.run_deepsense6g_gps_lidar_bgam --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--support-ratio`、`--label-space`、`--topk` 和 checkpoint 或 evaluation 相关参数

#### Scenario: 不新增顶层旧入口
- **WHEN** 架构边界测试扫描新 workflow
- **THEN** 测试 MUST 验证仓库根目录不存在新增的 `train_gps_lidar_bgam.py` 或 `eval_gps_lidar_bgam.py`
- **AND** 内部代码 MUST 不依赖顶层 `datasets.*`、`models.*` 或 `src.run_*` 入口

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 开发者执行 `import kd_sensing` 或导入配置/路径轻量模块
- **THEN** 系统 MUST 不因 BGAM workflow eager import torch dataset、LiDAR point cloud reader、matplotlib plotter 或训练 runtime
- **AND** BGAM 重依赖模块 MUST 只在对应 CLI、engine 或显式模块导入时加载

### Requirement: Hist 研究线不属于当前包结构
项目当前包结构 MUST 不再要求或暴露 HiST-Beam/Hist 专用 CLI、engine、model、evaluation 或 config 模块。`src/kd_sensing/engine` 与 `src/kd_sensing/models` MUST 保留当前主线职责模块，退役 Hist 专用文件后不得新增旧入口 facade。

#### Scenario: 包导入不要求 Hist 模块
- **WHEN** 开发者执行 `import kd_sensing`、`import kd_sensing.engine` 或 `import kd_sensing.models`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不要求 `kd_sensing.engine.hist_beam_*` 或 `kd_sensing.models.fusion.hist_beam` 存在

#### Scenario: 架构边界拒绝 Hist 旧入口
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 验证当前源码不再从 Hist 专用 engine/model/evaluation 模块导入运行逻辑
- **AND** 检查 MUST 验证没有新增 `hist_beam` 兼容聚合层

### Requirement: 退役研究线不触发本地产物隐式迁移
源码删除和包结构整理 MUST 与本地产物清理解耦。删除 Hist 源码 MUST 不自动移动、压缩或删除 `outputs/`、`logs/`、cache 或 checkpoint；本地产物删除 MUST 通过 runtime cleanup manifest 和显式删除阶段完成。

#### Scenario: 源码删除不隐式清理 outputs
- **WHEN** 实施者删除 Hist 源码、配置和文档入口
- **THEN** 该源码变更 MUST 不在同一步骤中用 ad hoc 命令删除 `outputs/`
- **AND** 需要删除的运行产物 MUST 先出现在 cleanup manifest 中

### Requirement: OpenSpec 当前规范不得保留脚手架占位
当前 `openspec/specs/` 中的 spec MUST 具备真实 Purpose 和可理解的需求文本。归档 change 产生的 `TBD`、空泛占位或未替换模板文本 MUST 在进入当前规范后被修复，架构边界测试 MUST 能发现这类漂移。

#### Scenario: 当前 spec purpose 可读
- **WHEN** 开发者运行架构边界检查或 OpenSpec hygiene 检查
- **THEN** 当前 specs 的 Purpose MUST 是描述 capability 边界的真实文本
- **AND** Purpose MUST 不包含 `TBD`、未替换模板提示或归档脚手架说明

#### Scenario: 新归档规范进入当前面
- **WHEN** 一个 change 被归档并生成或修改 `openspec/specs/` 下的当前 spec
- **THEN** 归档后的 spec MUST 通过 OpenSpec 校验和项目架构 hygiene 检查
- **AND** 若归档工具留下占位 Purpose，开发者 MUST 在同一清理批次修复

### Requirement: 架构 guardrail 必须匹配真实支持面
架构边界测试、inventory 文档和当前支持入口 MUST 使用同一套项目表面定义。新增、迁移或删除配置、脚本和公开入口时，项目 MUST 同步更新 guardrail、inventory 和引用文档，不得通过过宽阈值掩盖真实漂移。

#### Scenario: 配置数量 guardrail 更新
- **WHEN** `configs/fusion/` 的当前支持 YAML 集合发生变化
- **THEN** 架构边界测试中的数量阈值或 allowlist MUST 与 inventory 中的分类一致
- **AND** 测试 MUST 继续限制根目录无限增长

#### Scenario: 脚本 allowlist 更新
- **WHEN** shell orchestration、thin CLI alias 或 research diagnostic 脚本引用的配置路径变化
- **THEN** 脚本、inventory 和测试 allowlist MUST 同步更新
- **AND** 当前脚本 MUST 不引用不存在的配置文件作为默认入口

### Requirement: 大规模表面清理必须有快速验收
项目 MUST 为大规模表面清理提供快速验收命令，覆盖 OpenSpec 校验、架构边界、CLI help 和被修改入口的引用一致性。所有项目相关 Python 验收 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 清理实现后的快速验收
- **WHEN** 支持面清理实现完成
- **THEN** 开发者 MUST 运行 `openspec validate cleanup-project-surface-drift --strict`
- **AND** 开发者 MUST 运行 `openspec validate --all --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 修改 CLI 或脚本入口后验收
- **WHEN** 清理实现修改 console script、shell orchestration 或可视化入口
- **THEN** 开发者 MUST 运行对应 `--help` 或无副作用 smoke 检查
- **AND** 检查 MUST 不读取真实 dataset、不启动训练、不写入新的源码内产物

### Requirement: 退役失败实验路线不得保留源码支持面
当用户明确退役某条失败研究路线并要求不保留兼容时，项目 MUST 从当前源码支持面删除该路线的公开入口、配置、实现模块、测试和文档推荐路径。系统 MUST 不新增兼容 alias、stub CLI、薄 facade 或 registry fallback 来维持旧路线可发现性。

#### Scenario: 退役入口不可安装
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** 已退役路线的 `kd-sensing-*` 命令 MUST 不再由 `pyproject.toml` 声明
- **AND** 项目 MUST 不提供等价旧命令 alias 或兼容包装层

#### Scenario: 退役实现不可作为当前模块导入
- **WHEN** 开发者检查 `src/kd_sensing/cli`、`data`、`engine`、`models` 和 `losses`
- **THEN** 已退役路线专属模块 MUST 不再作为当前源码模块保留
- **AND** 保留主线不得从这些退役模块导入 helper

### Requirement: Top8 residual coarse 退役边界
Top8 selector 训练/plot/compare、GPS coarse anchor、GPS prior residual/delta correction、camera residual 和 Raymobtime s008 MUST 不属于当前包结构和推荐入口。BGAM、BGAM 依赖的 TopK candidate manifest/loss 支撑代码、通用 Top-K 指标、circular metrics、GPS-Rel-Polar、GPS v2、CSI、JEPA 和 viewer manifest MAY 保留；Raymobtime 旧名称只允许作为 migration guard 或退役说明出现。

#### Scenario: 保留通用指标
- **WHEN** 清理实现扫描到 `topk`、`candidate` 或 `residual` 字符串
- **THEN** 系统 MUST 按语义判断归属
- **AND** 普通 evaluation Top-K、viewer top-k 展示、CSI candidate ranking 和 GPS v2 自身 residual 诊断不得仅因字符串命中被删除

### Requirement: JEPA downstream 扩展实现边界
项目 MUST 将 JEPA Stage 1 预训练主模型、JEPA downstream pooler/adapter、模块化 conditioned encoder、optimizer 参数组和 runtime metadata 维护在职责清晰的窄模块中。新增 JEPA downstream pooler 或 adapter MUST 不要求修改 dataset、训练主循环、checkpoint schema 或旧兼容入口。

#### Scenario: 新增 JEPA pooler 不修改训练主循环
- **WHEN** 开发者新增一个 JEPA downstream pooler
- **THEN** 变更 MUST 限定在 JEPA downstream pooler/adapter 模块、注册代码、配置和测试
- **AND** 不需要修改 `engine.trainer` 主循环或 supervised beam loss/metric 流程

#### Scenario: 新增 JEPA adapter 不修改 dataset
- **WHEN** 开发者新增一个 JEPA downstream adapter
- **THEN** 变更 MUST 不要求修改 DeepSense6G dataset、GPS transform、image preprocessing 或 DataLoader 构建逻辑
- **AND** adapter MUST 通过模型配置和 registry 接入

#### Scenario: 不恢复退役入口
- **WHEN** JEPA downstream extensibility change 落地
- **THEN** 系统 MUST 不新增 KD/distillation、HiST/Hist、Top8 selector、GPS residual、camera residual 或 legacy fusion 兼容入口
- **AND** 新能力 MUST 通过当前 `src/kd_sensing` 包结构和 registry 边界接入

### Requirement: optimizer 参数组构建位于 optim 模块
训练引擎 MUST 将参数组解析、模块名 pattern 匹配、重复匹配检测、未匹配参数处理和参数组 summary 维护在 `kd_sensing.engine.optim` 或等价窄模块中。训练主循环 MUST 只消费构建好的 optimizer 和 summary。

#### Scenario: 修改 JEPA 参数组不触碰 trainer
- **WHEN** 开发者调整 JEPA context encoder、GPS encoder、pooler、core 或 head 的参数组匹配规则
- **THEN** 主要变更 MUST 限定在 optimizer 构建模块及其测试
- **AND** 不需要编辑 `engine.trainer` 的 epoch 或 batch 编排逻辑

#### Scenario: 参数组 summary 写入现有日志路径
- **WHEN** 训练使用多个 optimizer 参数组
- **THEN** 现有训练日志和 TensorBoard scalar 映射 MUST 能记录每组 learning rate 和参数数量
- **AND** 未声明参数组时 MUST 保持现有单 `main` 组日志字段

### Requirement: runtime metadata 收集位于 run metadata 模块
JEPA downstream 结构 metadata MUST 由 `engine.run_metadata`、artifact writer 或等价窄模块收集。模型和子模块 MAY 暴露只读 metadata 方法；训练主循环 MUST 不手写 JEPA downstream 专属字段。

#### Scenario: 模型声明 metadata 被聚合
- **WHEN** `model.primary` 或其子模块提供 JEPA downstream training strategy metadata
- **THEN** runtime metadata 收集模块 MUST 将其写入 `final_config.yaml` 或等价运行 metadata
- **AND** metadata MUST 包含 pooler、adapter、checkpoint、freeze 和参数组摘要中的正式字段

#### Scenario: config fallback 兼容历史配置
- **WHEN** metadata 在模型构建前需要从配置生成
- **THEN** run metadata 模块 MAY 使用配置解析作为 fallback
- **AND** fallback MUST 与模型声明 metadata 的核心字段保持一致

### Requirement: 项目健康护栏纳入架构边界
项目 MUST 将健康护栏纳入架构边界测试，使包结构、轻量导入、入口 allowlist、热点 inventory、测试 bootstrap 和分层验证命令保持一致。架构边界测试 MUST 能在全量 pytest 之前暴露项目支持面或维护性边界漂移。

#### Scenario: 架构边界检查健康护栏文件
- **WHEN** 开发者运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **THEN** 测试 MUST 验证 shared pytest bootstrap、项目表面积 inventory 和当前健康检查命令说明存在且互相一致
- **AND** 测试 MUST 拒绝新增未分类的长期脚本入口、未登记热点或明显回流的兼容 facade helper

#### Scenario: 新增热点必须声明边界
- **WHEN** 新代码引入超长 orchestration 函数、dataset 类、manifest builder 或兼容 facade
- **THEN** 变更 MUST 同步更新热点 inventory 或拆分到职责明确的窄模块
- **AND** 架构边界测试 MUST 提供可定位到文件和符号的失败信息

### Requirement: 测试基础设施不得重复项目路径注入
项目 MUST 将普通测试的源码路径注入集中管理。除架构边界 import probe、subprocess smoke 或明确隔离环境测试外，测试文件不得各自维护重复的 `ROOT/SRC/sys.path.insert` 启动逻辑。

#### Scenario: 普通测试文件不复制 bootstrap
- **WHEN** 开发者新增或修改普通单元测试
- **THEN** 测试 MUST 通过 shared pytest bootstrap 导入 `kd_sensing`
- **AND** 文件级 `sys.path.insert` 复制片段 MUST 被架构边界测试拒绝或要求显式例外说明

#### Scenario: 子进程边界测试可控
- **WHEN** 架构边界测试需要在干净解释器中验证某个 import 不牵出重依赖
- **THEN** 该测试 MAY 在子进程代码中显式设置 `sys.path`
- **AND** 该例外 MUST 保持局部，不得作为普通测试模板传播

### Requirement: 当前支持面收敛到 Image+GPS JEPA query-pool
项目 MUST 将当前推荐训练、评估、诊断和实验配置支持面收敛到 Image+GPS JEPA query-pool 主线及其必要对照。保留面 MUST 包含 `jepa_context_image + GPSQueryPool` JEPA downstream、`fair_gps_biased` paired baseline、supervised/random-best 控制组、vision-position baseline suite 和 `jepa_visual_analysis` 论文图/诊断出口。退役路线 MUST 不再作为 README 推荐入口、pyproject console script、架构 allowlist 或当前配置矩阵出现。

#### Scenario: README 展示当前主线
- **WHEN** 开发者阅读 README 的项目定位、主要入口和配置矩阵
- **THEN** 文档 MUST 把 Image+GPS JEPA query-pool、paired baseline/control、vision-position baseline suite 和 JEPA visual analysis 描述为当前主线
- **AND** 文档 MUST 不把 GPS window、DeepVerse/DT31、旧静态 modality visualization 或仓库级 Gradio viewer support 描述为当前入口
- **AND** 文档 MAY 继续保留 BeamBench/Arnold22 Camera AE+GPS Direct 当前入口和复现辅助说明

#### Scenario: 架构测试拒绝退役入口回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 拒绝退役的 viewer support、GPS window baseline、DeepVerse/DT31 workflow、Top8 selector dataset 和旧静态 modality visualization 文件重新出现在当前 allowlist 中
- **AND** 测试 MUST 继续允许 JEPA query-pool、paired control、vision-position baseline、BeamBench/Arnold22 Camera AE+GPS Direct 和 JEPA visual analysis 相关入口

#### Scenario: 配置矩阵只保留必要 JEPA 对照
- **WHEN** 开发者查看 `configs/fusion/experiments/jepa_image_gps/` 和实验矩阵文档
- **THEN** 当前配置 MUST 保留 query-pool、GPS-biased baseline、supervised baseline 和 random-best 控制组
- **AND** scene31-only、非 BeamBench 的 last-checkpoint 和 next-beam downstream ablation 配置 MUST 不再作为当前配置文件维护
- **AND** `beambench_fair` 相关配置 MAY 继续保留用于 Arnold22/BeamBench 口径对照

### Requirement: 退役 DeepVerse/DT31 数据生成路线
项目 MUST 不再维护 DeepVerse/DT31 数据生成、label builder、split、sanity check 或对应配置作为当前源码工作流。DeepVerse/DT31 的历史研究资料 MAY 留在非入口历史文档中，但 MUST 明确为退役背景，且 MUST 不再通过 registry、preprocess config、README quickstart 或架构 allowlist 暴露为当前 workflow。

#### Scenario: DeepVerse/DT31 源码入口不存在
- **WHEN** 开发者检查 `src/kd_sensing/data/deepverse/`、`configs/deepverse/` 和当前脚本入口 allowlist
- **THEN** DeepVerse/DT31 generator、label builder、split、sanity check 和 generation config MUST 不再作为源码入口存在
- **AND** 当前 README 和 inventory MUST 不推荐 DeepVerse/DT31 数据生成命令

#### Scenario: 不清理本地 DeepVerse 数据产物
- **WHEN** 本 change 删除 DeepVerse/DT31 源码和配置
- **THEN** 系统 MUST 不自动删除 `dataset/`、`outputs/`、`logs/`、cache 或 checkpoint 中的历史 DeepVerse 本地产物
- **AND** 如需清理本地产物，仍 MUST 使用 runtime cleanup manifest 工作流

### Requirement: 通用 baseline 与 workflow baseline 分层
项目 MUST 区分通用可训练 baseline 和 workflow/paper reproduction baseline。通用 baseline MUST 复用配置驱动训练、共享 batch/runtime 和模型 registry；workflow baseline MUST 只在需要官方协议、多阶段训练、特殊 metric 或报告产物时保留专用 orchestration，并 MUST 放在包内职责清晰的位置并记录生命周期、产物边界和 claim caveat。

#### Scenario: 通用 baseline 不修改训练循环
- **WHEN** 开发者新增普通 supervised/adaptation baseline
- **THEN** 变更 MUST 限定在配置、模型子组件、registry/default component 和 focused tests
- **AND** 不得为了该 baseline 修改 dataset 解析、训练主循环或公共 CLI 入口

#### Scenario: 论文复现 workflow 有边界
- **WHEN** 开发者新增包含官方协议、多阶段训练、特殊 metrics 或报告产物的 workflow baseline
- **THEN** 代码 MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或明确生命周期的薄 alias
- **AND** 文档 MUST 标记其不是普通 `modular_sequence` baseline，并说明输出只写入 ignored runtime artifact root

### Requirement: 新模型不得扩大入口表面
新增模型架构能力 MUST 不新增 root-level 旧脚本、兼容聚合层、退役研究线实体配置或绕过 `src/kd_sensing` 包结构的运行方式。若需要新增 CLI，MUST 是包内 console script 或 lifecycle 登记的薄 alias，并同步 pyproject、README/docs、inventory 和架构边界测试。

#### Scenario: 新模型需要命令入口
- **WHEN** whole-model exception 或 workflow baseline 需要新的用户命令
- **THEN** 入口 MUST 通过包内 CLI 或登记的薄 alias 暴露
- **AND** 系统 MUST 不新增仓库根长期训练脚本或未登记脚本入口

