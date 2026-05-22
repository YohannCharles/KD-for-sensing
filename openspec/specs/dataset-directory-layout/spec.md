# dataset-directory-layout Specification

## Purpose
定义本地数据集家族目录布局、可生成数据边界和路径解析约束。
## Requirements
### Requirement: 数据集家族目录规范
项目 MUST 在 `dataset/` 下按数据集家族组织本地数据。每个数据集家族 MUST 拥有独立的一级目录，场景、天气或条件等细分 MUST 放在对应家族目录内部。

#### Scenario: DeepSense6G 使用家族目录
- **WHEN** 用户使用未显式覆盖 `data_root` 的 DeepSense6G 配置
- **THEN** 默认数据根目录 MUST 位于 `dataset/DeepSense6G/` 下
- **AND** Scenario 31、Scenario 32 和 Scenario 9 MUST 分别使用 `scenario31`、`scenario32` 和 `scenario9` 子目录

#### Scenario: 数据集家族平级
- **WHEN** 项目同时存在 DeepSense6G 和 MMW 本地数据
- **THEN** DeepSense6G 数据 MUST 位于 `dataset/DeepSense6G/`
- **AND** MMW 数据 MUST 位于 `dataset/MMW/`

### Requirement: Dataset layout descriptor
项目 MUST 提供集中式 dataset layout descriptor 或等价机制，统一描述数据集家族、规范根目录、旧兼容根目录、场景/条件别名和必要子目录。训练、评估和预处理代码 MUST 复用该机制，不得在多个入口重复硬编码同一默认目录规则。

#### Scenario: 解析 DeepSense6G 规范根目录
- **WHEN** 代码请求 DeepSense6G Scenario 31 的默认根目录
- **THEN** layout descriptor MUST 返回 `dataset/DeepSense6G/scenario31`
- **AND** 返回结果 MUST 可被现有 `resolve_path` 按项目根目录解析

#### Scenario: 记录旧兼容根目录
- **WHEN** 代码请求 DeepSense6G Scenario 31 的 legacy 根目录
- **THEN** layout descriptor MUST 能提供 `dataset/scenario31`
- **AND** legacy 根目录 MUST 只用于显式覆盖、迁移文档或兼容性校验，不得作为未显式配置时的默认值

### Requirement: MMW 天气目录规范
未来 MMW 数据 MUST 按天气条件组织为 `dataset/MMW/<condition>/Sensor_Data` 和 `dataset/MMW/<condition>/Channel_Data`。首批条件名 MUST 包含 `sunny`、`rainy` 和 `foggy`。

#### Scenario: sunny MMW 目录
- **WHEN** 用户准备 sunny 条件的 MMW 数据
- **THEN** 传感器数据 MUST 放入 `dataset/MMW/sunny/Sensor_Data`
- **AND** 信道数据 MUST 放入 `dataset/MMW/sunny/Channel_Data`

#### Scenario: rainy 和 foggy MMW 目录
- **WHEN** 用户准备 rainy 或 foggy 条件的 MMW 数据
- **THEN** 对应条件目录 MUST 分别为 `dataset/MMW/rainy` 或 `dataset/MMW/foggy`
- **AND** 每个条件目录 MUST 包含 `Sensor_Data` 与 `Channel_Data` 两个语义子目录

### Requirement: 数据文件不自动迁移
代码变更 MUST 不自动移动、复制或删除 `dataset/` 下的真实数据文件。目录迁移 MUST 由用户显式执行，或通过显式配置 `data_root` 继续使用旧目录。

#### Scenario: 默认配置不移动旧数据
- **WHEN** 用户运行默认 DeepSense6G 配置且本地存在旧 `dataset/scenario31`
- **THEN** 系统 MUST 不自动移动该目录
- **AND** 系统 MUST 不删除、覆盖或复制旧目录中的文件

#### Scenario: 显式旧路径继续可用
- **WHEN** 用户设置 `data.dataset.data_root: dataset/scenario31`
- **THEN** 系统 MUST 使用该显式路径构建 dataset
- **AND** 该行为 MUST 不依赖 `dataset/DeepSense6G/scenario31` 是否存在

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
