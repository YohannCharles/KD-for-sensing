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
