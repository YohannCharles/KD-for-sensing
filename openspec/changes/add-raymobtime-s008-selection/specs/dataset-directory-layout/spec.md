## ADDED Requirements

### Requirement: Raymobtime 数据集家族目录规范
项目 MUST 将 Raymobtime 本地数据作为独立数据集家族放在 `dataset/Raymobtime/` 下。Raymobtime s008 的默认规范根目录 MUST 为 `dataset/Raymobtime/s008`，并 MUST 保留用户显式传入外部 `data_root` 的能力。

#### Scenario: Raymobtime s008 默认目录
- **WHEN** 用户使用 `data.dataset.type: raymobtime_s008` 且未显式配置 `data.dataset.data_root`
- **THEN** dataset layout descriptor MUST 返回 `dataset/Raymobtime/s008`
- **AND** 返回路径 MUST 可被现有项目根路径解析工具解析

#### Scenario: Raymobtime 与现有数据集家族平级
- **WHEN** 项目同时存在 DeepSense6G、MMW 和 Raymobtime 本地数据
- **THEN** DeepSense6G 数据 MUST 位于 `dataset/DeepSense6G/`
- **AND** MMW 数据 MUST 位于 `dataset/MMW/`
- **AND** Raymobtime 数据 MUST 位于 `dataset/Raymobtime/`

#### Scenario: 显式外部 data_root
- **WHEN** 用户配置 `data.dataset.type: raymobtime_s008` 且显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用该显式路径构建 dataset 或预处理任务
- **AND** 系统 MUST 不自动移动、复制或删除该路径下的真实数据文件

### Requirement: Raymobtime s008 本地产物边界
Raymobtime s008 的原始数据、cache、审计报告、训练输出、日志和 checkpoint MUST 继续遵守源码与本地产物边界。项目文档和配置 MUST 不要求提交这些本地产物。

#### Scenario: Raymobtime cache 默认不提交
- **WHEN** 用户运行 Raymobtime s008 预处理并生成 cache
- **THEN** cache MUST 默认写入 `outputs/`、`dataset/Raymobtime/s008/cache` 或用户配置的 ignored 目录
- **AND** 项目文档 MUST 标记这些文件为本地产物

#### Scenario: 不自动迁移 Raymobtime_s008 旧目录
- **WHEN** 用户本地已有 `Raymobtime_s008/` 或其它外部数据目录
- **THEN** 系统 MUST 不自动把该目录移动到 `dataset/Raymobtime/s008`
- **AND** 用户 MUST 能通过显式 `data_root` 继续使用该目录
