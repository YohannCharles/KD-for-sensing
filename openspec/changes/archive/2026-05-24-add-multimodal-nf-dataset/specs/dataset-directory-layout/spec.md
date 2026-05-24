## ADDED Requirements

### Requirement: Multimodal-NF 数据集家族目录规范
项目 MUST 将 Multimodal-NF 本地数据作为独立数据集家族放在 `dataset/MultimodalNF/` 下。该家族目录 MUST 区分用户放置的真实原始数据、codebook 文件、可生成 cache 和输出报告，并 MUST 保留显式外部路径覆盖能力。

#### Scenario: Multimodal-NF 默认目录
- **WHEN** 用户使用 `data.dataset.type: multimodal_nf` 且未显式配置 `data.dataset.data_root`
- **THEN** dataset layout descriptor MUST 返回 `dataset/MultimodalNF`
- **AND** 返回路径 MUST 可被现有项目根路径解析工具解析

#### Scenario: Multimodal-NF 与现有数据集家族平级
- **WHEN** 项目同时存在 DeepSense6G、MMW、Raymobtime 和 Multimodal-NF 本地数据
- **THEN** DeepSense6G 数据 MUST 位于 `dataset/DeepSense6G/`
- **AND** MMW 数据 MUST 位于 `dataset/MMW/`
- **AND** Raymobtime 数据 MUST 位于 `dataset/Raymobtime/`
- **AND** Multimodal-NF 数据 MUST 位于 `dataset/MultimodalNF/`

### Requirement: Multimodal-NF 本地产物边界
Multimodal-NF 的 HDF5、zip、codebook、cache、审计报告、训练输出、日志和 checkpoint MUST 遵守源码与本地产物边界。项目文档和配置 MUST 不要求提交这些本地产物。

#### Scenario: 原始数据和 codebook 不提交
- **WHEN** 用户下载 Hugging Face Multimodal-NF 数据、官方 HDF5、image/lidar zip 或 `upa64x64_NF_codebook*.pkl`
- **THEN** 这些文件 MUST 被视为本地数据输入
- **AND** 项目文档 MUST 明确它们通常不进入源码变更

#### Scenario: cache 默认不提交
- **WHEN** 用户运行 Multimodal-NF 审计、index 构建或 cache 构建
- **THEN** 生成文件 MUST 默认写入 `outputs/`、`dataset/MultimodalNF/cache` 或用户配置的 ignored 目录
- **AND** 系统 MUST 不要求将 cache、审计报告、训练日志或 checkpoint 纳入源码

### Requirement: Multimodal-NF 数据文件不自动迁移
代码变更 MUST 不自动移动、复制、删除或解压 `dataset/MultimodalNF/` 或显式外部路径下的真实数据文件。目录迁移和解压 MUST 由用户显式执行，或通过显式配置指定。

#### Scenario: 显式外部 data_root
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf` 且显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用该路径构建 audit、index 或 dataset
- **AND** 系统 MUST 不自动把该路径内容移动到 `dataset/MultimodalNF`

#### Scenario: zip 文件处理
- **WHEN** 用户提供 image 或 LiDAR zip 路径
- **THEN** 系统 MUST 只在显式预处理配置允许时读取或解压
- **AND** 解压输出 MUST 位于用户配置的本地产物目录
