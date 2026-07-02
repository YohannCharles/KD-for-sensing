# raymobtime-s008-retirement Specification

## Purpose
记录 Raymobtime s008 工作流的退役状态、清理边界和引用归零要求，确保当前源码、配置、文档、测试和注册入口不再把 Raymobtime s008 作为可运行支持面暴露。
## Requirements
### Requirement: Raymobtime s008 退役状态
项目 MUST 将 Raymobtime s008 视为已退役工作流。源码、配置、文档、测试和注册入口 MUST 不再把 `raymobtime_s008`、Raymobtime s008 预处理器、Raymobtime s008 selection 模型或 Raymobtime s008 数据目录作为当前支持能力暴露。

#### Scenario: 旧 Raymobtime 配置快速失败
- **WHEN** 用户加载 `data.dataset.type: raymobtime_s008`、Raymobtime s008 预处理配置或 Raymobtime s008 selection 模型配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含 Raymobtime s008 已退役
- **AND** 系统 MUST 不导入 Raymobtime s008 dataset、model 或 preprocessing 实现

#### Scenario: 当前支持面不列出 Raymobtime
- **WHEN** 开发者查询当前 dataset descriptor、preprocessor registry、model registry、推荐配置或 README 快速入口
- **THEN** 系统 MUST 不把 Raymobtime s008 列为可运行工作流
- **AND** DeepSense6G、MMW、CSI、viewer 和通用训练/评估/预处理入口 MUST 保持可用

### Requirement: Raymobtime s008 清理清单
删除 Raymobtime s008 本地数据或运行产物前，系统 MUST 生成 machine-readable manifest。manifest MUST 记录每个候选路径、匹配原因、产物类型、大小、是否在项目根内，以及是否会被删除。

#### Scenario: 生成清理 manifest
- **WHEN** 用户要求删除 Raymobtime s008 数据集和相关产物
- **THEN** 清理流程 MUST 先写出 manifest
- **AND** manifest MUST 至少覆盖 `dataset/Raymobtime/s008`、Raymobtime s008 cache、审计报告、训练输出、评估输出、日志和 checkpoint 候选
- **AND** manifest MUST 标明不存在路径为 skipped

#### Scenario: 清理边界限制
- **WHEN** 清理流程执行删除
- **THEN** 系统 MUST 只删除 manifest 中确认为 Raymobtime s008 且位于允许边界内的候选路径
- **AND** 系统 MUST NOT 删除 DeepSense6G、MMW、CSI、`All_models/` 已跟踪权重、OpenSpec artifacts、源码以外未知外部 `data_root` 或其它活跃实验产物

### Requirement: Raymobtime s008 引用归零
实现完成后，源码、配置、文档和测试中的 Raymobtime s008 支持性引用 MUST 被删除或改为退役说明。保留的历史 archive 内容 MAY 继续存在，但 MUST 不作为当前运行入口或健康检查依据。

#### Scenario: 当前代码引用扫描
- **WHEN** 开发者扫描当前源码、配置、README、docs 和 tests
- **THEN** 不得存在导入、注册、默认配置或推荐命令继续要求 Raymobtime s008 可运行
- **AND** 若保留历史说明，文本 MUST 明确 Raymobtime s008 已退役

#### Scenario: 测试矩阵更新
- **WHEN** 开发者运行快速健康检查说明或测试脚本
- **THEN** 检查 MUST 不要求 `tests/test_raymobtime_s008_selection.py` 存在或通过
- **AND** 检查 MUST 覆盖 Raymobtime s008 退役错误、通用 CLI help 和当前保留 workflow 的导入边界

### Requirement: Raymobtime s008 workflow 已退役
Raymobtime s008 预处理、训练、评估、smoke 和实验矩阵 workflow 已退役，不属于当前实验入口。旧 `configs/raymobtime/*`、`configs/preprocess/raymobtime_s008_*.yaml`、`raymobtime_s008` dataset/model/preprocessor 名称和 selection 模型名称 MUST 只作为 migration guard 命中或历史说明出现。

#### Scenario: 旧 Raymobtime 预处理配置被拒绝
- **WHEN** 用户运行 `kd-sensing-preprocess` 并引用 Raymobtime s008 预处理配置或 preprocessor type
- **THEN** 系统 MUST fail fast
- **AND** 错误信息 MUST 明确 Raymobtime s008 已退役且无兼容迁移入口

#### Scenario: 旧 Raymobtime 训练配置被拒绝
- **WHEN** 用户运行退役历史命令 `conda run -n kd_mm_beam kd-sensing-train --config configs/raymobtime/s008_multitask_selection.yaml`
- **THEN** 系统 MUST 不构建 `raymobtime_s008` dataset、selection 模型或 Raymobtime cache
- **AND** 错误信息 MUST 指向当前保留 workflow 或说明该研究线已退役

#### Scenario: Raymobtime smoke 与矩阵不作为当前要求
- **WHEN** 开发者运行当前架构边界、config load 或实验 workflow 测试
- **THEN** 测试 MUST 不要求 Raymobtime dataset smoke、训练 smoke、评估 smoke 或推荐实验矩阵存在
- **AND** 若测试覆盖 Raymobtime 名称，MUST 只验证 migration guard 或 registry 拒绝语义

### Requirement: 推荐实验工作流不包含 Raymobtime s008
README、实验矩阵、快速健康检查和配置驱动 workflow 文档 MUST 不再把 Raymobtime s008 作为当前推荐或可运行实验。历史 archive MAY 保留 Raymobtime 记录，但 MUST 不作为当前入口、教程或验证命令。

#### Scenario: README 和实验矩阵移除 Raymobtime 入口
- **WHEN** 用户阅读 README、docs/experiment_matrix.md 或研究笔记中的当前推荐流程
- **THEN** 文档 MUST 不再推荐运行 Raymobtime s008 预处理、训练或评估命令
- **AND** 文档 MUST 明确当前主线使用仍保留的数据集和 viewer workflow

#### Scenario: 健康检查不要求 Raymobtime focused test
- **WHEN** 开发者执行快速验证说明中的 focused tests
- **THEN** 验证命令 MUST 不要求 `tests/test_raymobtime_s008_selection.py`
- **AND** 验证 MUST 覆盖通用 CLI、架构边界、配置退役 guard 和当前保留 workflow

#### Scenario: 旧 Raymobtime 配置不可作为 workflow
- **WHEN** 用户传入 `configs/raymobtime/` 或 `configs/preprocess/raymobtime_s008_*.yaml` 下的旧配置路径
- **THEN** 系统 MUST 拒绝该 workflow 或这些配置文件 MUST 已被删除
- **AND** 错误信息 MUST 指出 Raymobtime s008 已退役

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
