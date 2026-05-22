## ADDED Requirements

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
