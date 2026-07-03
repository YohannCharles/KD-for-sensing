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
项目 MUST 使用 `pyproject.toml` 声明的 `kd-sensing-*` package console scripts 或包内 CLI module 作为当前支持的运行入口。`scripts/*.py` 中只转发到包内 CLI 的 Python thin alias MUST 从当前支持面删除，不得作为 README、AGENTS、docs、OpenSpec 或维护索引推荐入口。项目 MUST 继续删除现有顶层旧脚本入口，包括 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 和 `gen_data_seq.py`，不得保留兼容包装脚本。

#### Scenario: 运行训练 console script 帮助信息
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-train --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 展示配置文件、训练任务和命令行覆盖相关的参数说明

#### Scenario: 运行评估和预处理 console script 帮助信息
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-evaluate --help` 或 `conda run -n kd_mm_beam kd-sensing-preprocess --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 展示对应任务的参数说明

#### Scenario: 旧脚本入口和 thin alias 已删除
- **WHEN** 结构收敛完成后检查仓库根目录和 `scripts/`
- **THEN** 根目录 MUST 不存在 `train_image.py`、`train_both.py`、`test_model_image.py`、`test_model_both.py`、`CSV_process.py` 或 `gen_data_seq.py`
- **AND** `scripts/` MUST 不保留 `train.py`、`evaluate.py`、`preprocess.py`、`check_dataset.py`、`eval_baseline.py`、`train_baseline.py`、`train_beambench_image_ae_gps.py` 或 `run_beambench_image_ae_gps_tableiii.py` 这类 Python thin alias

#### Scenario: 文档不推荐 thin alias
- **WHEN** 开发者阅读 README、AGENTS、`docs/agent_navigation.md`、维护索引或当前 OpenSpec specs 中的运行入口说明
- **THEN** 当前训练、评估、预处理和 BeamBench workflow MUST 指向 `kd-sensing-*` console scripts 或明确包内 CLI
- **AND** 文档 MUST 不把已删除的 `scripts/*.py` thin alias 写成当前推荐命令

### Requirement: 项目根路径与资源路径统一
项目 MUST 提供统一路径解析工具，用于解析项目根目录、数据目录、权重目录、配置目录和输出目录。运行入口 MUST 通过该工具解析相对路径，避免依赖当前工作目录偶然匹配。

#### Scenario: 从子目录运行命令
- **WHEN** 开发者从仓库子目录调用新脚本或新 CLI，并传入相对数据路径
- **THEN** 系统 MUST 根据项目根路径解析资源位置，而不是错误地相对于当前子目录查找

#### Scenario: 读取默认资源目录
- **WHEN** 用户未显式传入数据目录或权重目录
- **THEN** 系统 MUST 使用配置中的默认路径，并能定位当前仓库的 `dataset/` 和 `All_models/` 目录

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

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的 console script MUST 是 parser/config glue，不得复制长期维护的 parser 或主实现。项目 MUST 不再要求安装 `kd-sensing-raymobtime-analysis`、GPS window baseline、viewer manifest 或仓库级 Gradio viewer support 入口。BeamBench 相关 console scripts MAY 保持当前声明。

#### Scenario: 退役 viewer scripts 不声明
- **WHEN** 开发者检查 `pyproject.toml` entry points
- **THEN** 项目 MUST 不声明 `kd-sensing-export-viewer-manifest`
- **AND** 项目 MUST 不声明 `kd-sensing-visualize-modalities`
- **AND** 项目 MUST 不声明仓库级 Gradio viewer support 入口

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-jepa-visual-analysis` 和 `kd-sensing-jepa-gps-shortcut-benchmark`
- **AND** 安装生成的 entry points MUST 不要求包含 `kd-sensing-raymobtime-analysis`、`kd-sensing-gps-window-baseline`、viewer manifest 或仓库级 Gradio viewer support 入口

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

### Requirement: Distillation 子包已移除
项目 MUST 不再要求或暴露 `kd_sensing.distillation` 子包。监督 soft target、beam smoothing、adapter/prototype loss 和其它当前训练方法 MUST 通过 `losses`、objective 或 training extension 表达。

#### Scenario: distillation 子包导入不可作为要求
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 不要求 `kd_sensing.distillation` 可导入
- **AND** 默认组件导入 MUST 不导入 distiller registry

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

### Requirement: 入口生命周期必须可审计
项目 MUST 为 `scripts/` 和 `tools/analysis/` 中保留的入口维护生命周期分类。新增或保留入口 MUST 属于 package CLI、研究诊断脚本、数据准备脚本、config generator、figure helper 或 local/manual validation/runner 中的一类，并在架构检查 allowlist 或 inventory 中记录原因。固定 GPU queue shell 已退役；仓库级 `tools/visualization/` viewer support 也 MUST 不再作为当前入口分类回流。

#### Scenario: 新增脚本入口需要分类
- **WHEN** 开发者新增 `scripts/` 或 `tools/analysis/` 下的 Python 或 shell 入口
- **THEN** 架构边界检查 MUST 要求该入口出现在生命周期 allowlist 或 inventory 中
- **AND** 如果该入口复制已有 console script 或包内 CLI 工作流，检查 MUST 拒绝该入口

#### Scenario: 重复 wrapper 不作为推荐入口
- **WHEN** 包内 CLI 或 console script 已覆盖训练、评估、预处理或当前诊断工作流
- **THEN** README 和工具文档 MUST 推荐包内 CLI 或 console script
- **AND** 对应 `scripts/` 或 `tools/` fallback wrapper MUST 删除或被明确标注为短期研究/数据准备入口

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

### Requirement: CLI 与实现模块职责分离
项目 SHALL 保持 CLI/脚本入口与真实 workflow 实现的职责分离。Package CLI、保留的研究诊断脚本、数据准备脚本、config generator 和 local/manual helper MUST 只承担参数解析、配置覆盖、轻量 IO、调用包内实现和 user-facing exit code；训练、评估、benchmark、dataset preparation 或诊断主逻辑 MUST 位于对应职责模块。

#### Scenario: package CLI 调用 owner module
- **WHEN** 新增或修改 package console script
- **THEN** CLI 文件 MUST 调用 `baselines/`、`diagnostics/`、`engine/`、`data/` 或其它对应 owner module 中的实现
- **AND** CLI 文件 MUST 不复制通用训练循环、评估循环、模型 forward 分支或 dataset parsing 主逻辑

#### Scenario: scripts 不恢复 Python thin alias
- **WHEN** 新增或保留 `scripts/` 下的 Python thin alias
- **THEN** 架构边界测试 MUST 失败
- **AND** 对应入口 MUST 改为 package console script、包内 CLI 或直接移除

### Requirement: 内部 helper 合并与边界保留
项目 MUST 允许同一 owner 下的内部 helper 在过度拆分、单调用点、低复用或只服务 re-export 时合并回清晰 owner 模块。合并 MUST 不新增旧入口、兼容聚合层、仓库根运行方式或跨领域 `utils` 聚合模块。合并后 public facade、console script、轻量导入边界、数据/配置/训练职责边界和本地产物边界 MUST 保持稳定。

#### Scenario: 合并同 owner 内部 helper
- **WHEN** 开发者将只被同一 owner 模块使用的内部 helper 文件合并回该 owner 模块
- **THEN** 包内公开 import、CLI 入口和 console script MUST 继续指向同一 public surface
- **AND** 合并后的实现 MUST 不要求调用方从旧 helper 文件导入符号
- **AND** 架构边界测试和治理索引 MUST 更新为合并后的 owner 文件布局

#### Scenario: 不创建新的兼容聚合层
- **WHEN** 开发者为了减少 Python 文件数调整模块布局
- **THEN** 系统 MUST 不新增只转发旧路径的兼容 facade、跨领域 `helpers.py`、仓库根脚本入口或绕过 `src/kd_sensing` 包结构的运行方式
- **AND** 已退役入口和 retired research line MUST 不因合并而恢复

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 合并发生在 diagnostics、engine、preprocessing 或 baseline 内部模块
- **THEN** 导入轻量配置、路径工具、包级公共 API 或已登记 thin facade 时 MUST 不额外触发 dataset 读取、模型权重加载、训练逻辑或重型可视化依赖

### Requirement: 项目架构右尺寸化必须基于 owner 职责而非全局数量
项目 MUST 使用 owner 职责、公开 surface、导入边界、热点预算、复用关系和验证覆盖来判断模块是否应拆分、合并或保留。Python 文件数、function 数和 import 数 MUST 作为架构审计基线和趋势信号，但 MUST NOT 单独作为要求合并或拆分的硬性目标。

#### Scenario: 文件数较多但职责清晰
- **WHEN** 架构审计发现某个区域存在较多 Python 文件
- **THEN** 审计 MUST 先判断这些文件是否对应独立 owner、thin CLI、focused tests、轻量导入边界或公开兼容 facade
- **AND** 系统 MUST NOT 仅因为文件数量高就合并这些模块

#### Scenario: 文件较少但函数过长
- **WHEN** 架构审计发现单个 owner 中存在超预算 orchestration 函数、长初始化函数或混合 schema/write/runtime 职责的实现
- **THEN** 系统 MUST 将其登记为 hotspot、monitor 或 split-next
- **AND** 拆分方向 MUST 指向稳定职责边界，而不是按固定行数机械切割

### Requirement: 热点拆分必须保持公开行为和本地产物边界兼容
热点模块拆分 MUST 只改变内部模块组织，不得改变公开 CLI 名称、console scripts、public import owner、配置路径、数据 split 语义、beam label 语义、指标口径、manifest schema、run metadata、默认输出路径或本地产物边界。

#### Scenario: 拆分公开 workflow owner
- **WHEN** 开发者拆分 BeamBench、trainer、dataset、diagnostics 或 benchmark owner
- **THEN** 包内公开 import、CLI 入口和 console script MUST 继续指向同一 public surface
- **AND** focused tests MUST 覆盖该 workflow 的关键 schema、summary、metadata 或 metric 输出

#### Scenario: streamlining wave 后 owner 边界稳定
- **WHEN** 开发者继续修改 dataset family、training runtime、evaluation pass、modular model forward、diagnostics runner 或 MMW GPS v2 workflow
- **THEN** 变更 MUST 优先落在已登记的窄 owner、run context、family adapter、stage helper 或 artifact writer 中
- **AND** 不得把 condition layout、checkpoint restore、batch evaluation、forward diagnostics、protocol dispatch 或 artifact schema 写回公开 facade 或单个巨型入口函数

#### Scenario: 拆分不触碰本地产物
- **WHEN** 开发者实施热点拆分或运行对应验证
- **THEN** 变更 MUST NOT 删除、移动、重写或提交 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史本地产物
- **AND** 新增临时验证产物 MUST 位于测试临时目录或已忽略的本地产物路径

### Requirement: 同 owner 低价值 helper 可以合并但不得恢复兼容聚合层
项目 MUST 允许将同一 owner 下单调用点、只服务 re-export、无独立 public contract、无复用价值或仅为降低行数而产生的 helper 合并回清晰 owner 模块。合并 MUST 不新增旧入口、跨领域 `helpers.py`、兼容聚合层、仓库根运行方式或退役研究线入口。

#### Scenario: 合并内部 helper
- **WHEN** 开发者合并一个只被同一 owner 使用的内部 helper 文件
- **THEN** 调用方 MUST 继续使用 owner 的公开 import 或已登记窄模块
- **AND** 架构边界测试或治理索引 MUST 更新为合并后的模块布局

#### Scenario: 禁止用兼容 wrapper 降低迁移成本
- **WHEN** helper 文件被合并或删除
- **THEN** 系统 MUST NOT 新增只转发旧 helper 路径的兼容 wrapper
- **AND** 内部代码 MUST NOT 从公开 facade 回流导入 suite-specific helper

### Requirement: Architecture streamlining campaign preserves public behavior
项目 MUST 允许按 wave 执行全仓架构收敛，但该收敛 MUST 只改变内部模块组织和未登记 public surface 的 import 路径，不得隐式改变当前 package CLI 名称、current canonical config 语义、dataset split 语义、beam label / label-space 口径、metric schema、checkpoint schema、run metadata 字段、默认输出分区或本地产物边界。

#### Scenario: 用户可见入口保持稳定
- **WHEN** architecture streamlining wave 修改 `data`、`engine`、`models`、`diagnostics`、`config`、`scripts` 或 `configs`
- **THEN** 当前 README、pyproject console scripts、current OpenSpec specs 和 inventory 登记的 package CLI MUST 继续可用
- **AND** 已登记 current workflow 的用户可见输入/输出契约 MUST 保持兼容

#### Scenario: 内部结构可以 breaking 收缩
- **WHEN** 某个 import path 未被 README、pyproject console script、current spec、inventory public surface 或 focused test 明确登记为 public owner
- **THEN** 该 path MAY 在本 change 中被删除、合并或迁到真实 owner
- **AND** 内部调用方 MUST 改为导入职责明确的 owner module，不得新增兼容 wrapper 维持旧路径

### Requirement: Architecture streamlining starts from a clean or documented baseline
项目 MUST 在实施任何源码 wave 前记录工作树、active change 和验证 baseline。若工作树存在无关实验改动、未跟踪配置/脚本、本地 cache 噪声或已完成但未归档的 active change，实施说明 MUST 先归档、提交、隔离，或明确记录 deferral 和影响范围。

#### Scenario: Wave 0 captures baseline state
- **WHEN** 本 change 进入 implementation
- **THEN** tasks 或实现说明 MUST 记录 `openspec list --json`、`git status --short`、已知未跟踪实验表面、产物边界占位文件状态和 baseline validation 命令结果
- **AND** 后续源码 wave MUST 不把无关实验变更或本地运行产物混入架构重构 diff

#### Scenario: 已完成 active change 不被误用
- **WHEN** active change 显示 status 为 complete
- **THEN** 本 change MUST 先归档该 change，或在 Wave 0 中说明暂不归档的原因、风险和与本 change 的隔离方式

### Requirement: Surface pruning preserves current user behavior
项目 MAY 大规模删除旧入口、本地脚本、隐藏 CLI、重复 tombstone 和可生成配置，但 MUST 保持 current package CLI、current canonical config、dataset split、beam label/label-space、metric schema、checkpoint schema、run metadata 和默认本地产物分区兼容。

#### Scenario: Current public behavior unchanged
- **WHEN** 本 change 删除或合并 internal surface
- **THEN** README、pyproject console scripts、current specs 和 inventory 登记的 current workflow MUST 继续可用
- **AND** 删除 MUST 不要求用户改用未记录的新命令

#### Scenario: Internal breaking import allowed
- **WHEN** 一个 import path 未登记为 public surface
- **THEN** 它 MAY 被删除或移动
- **AND** 项目 MUST 不新增旧路径 compatibility wrapper

