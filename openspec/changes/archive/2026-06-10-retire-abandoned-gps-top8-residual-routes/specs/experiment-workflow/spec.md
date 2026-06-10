## ADDED Requirements

### Requirement: 当前推荐 workflow 排除 Top8 residual coarse 路线
README、实验矩阵、quickstart、docs inventory 和健康检查 MUST 不再把 Top8 selector、standalone Top8 candidate manifest、GPS coarse anchor、GPS prior residual correction 或 camera residual 描述为当前可运行或推荐 workflow。当前推荐面 MUST 聚焦仍保留的 supervised/adaptation、GPS v2/adapter、MMW GPS v2、BGAM、CSI hardening、Raymobtime、JEPA、预处理、诊断和 viewer manifest。

#### Scenario: quickstart 不展示退役命令
- **WHEN** 开发者阅读 README、README_REPRODUCE 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不提供退役 Top8 selector/residual/GPS coarse anchor 命令作为当前运行步骤
- **AND** 文档 MUST 指向仍存在的配置化 CLI 和保留 workflow

#### Scenario: 健康检查不要求退役入口
- **WHEN** 开发者执行快速健康检查或架构边界测试
- **THEN** 检查 MUST 不要求退役 console scripts、配置、CLI、engine、model 或 loss 可导入
- **AND** 检查 MAY 断言这些入口已不存在

## MODIFIED Requirements

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态/少样本跨场景 beam prediction、DeepSense6G/MMW/Raymobtime supervised/adaptation、GPS v2/adapter、MMW Town GPS v2、BGAM、CSI hardening、Raymobtime s008 selection、JEPA、预处理、诊断和 viewer manifest，而不是 KD-first、HiST-Beam-first、Top8/residual-first 或 GPS coarse-anchor-first 工作流。历史 KD、Hist、Top8 selector、residual、camera residual 或 GPS coarse anchor 背景可以保留在 archive 或历史说明中，但必须标记为已退役或历史记录。

#### Scenario: pyproject 描述不再 KD Hist 或退役路线 first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation、HiST-Beam、Top8 selector、residual 或 GPS coarse anchor 描述为当前唯一或首要工作流
- **AND** 若提到这些路线，MUST 表达其为 legacy、historical 或 retired

#### Scenario: 文档保留历史说明
- **WHEN** README 或 docs 提到历史 KD、Hist、Top8 selector、residual、camera residual 或 GPS coarse anchor 代码
- **THEN** 文档 MUST 说明对应能力已从当前 active mainline 退役
- **AND** 文档 MUST 不提供当前推荐运行命令

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、DeepSense6G GPS v2/adapter、MMW Town GPS v2、BGAM、CSI hardening、Raymobtime s008 selection、JEPA、预处理、诊断和 viewer manifest。KD baseline、HiST-Beam/Hist、Top8 selector、standalone Top8 candidate manifest、GPS coarse anchor、residual fusion、camera residual、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、retired Top8 selector/residual/GPS coarse anchor 命令或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、Top8 selector、residual、camera residual、GPS coarse anchor、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/Raymobtime 主线

## REMOVED Requirements

### Requirement: Residual workflow query leakage guard
**Reason**: DeepSense6G residual workflow 已退役。
**Migration**: 无兼容迁移；保留 workflow 自行定义 query leakage guard。

### Requirement: Residual workflow result contract
**Reason**: DeepSense6G residual 输出契约随 workflow 退役。
**Migration**: 无兼容迁移。

### Requirement: Residual workflow acceptance commands
**Reason**: residual CLI 和配置将删除，不再有验收命令。
**Migration**: 使用保留 workflow 的 focused validation。

### Requirement: Camera residual staged CLI workflow
**Reason**: camera residual staged workflow 已退役。
**Migration**: 无兼容迁移。

### Requirement: Camera residual query leakage guard
**Reason**: camera residual training/evaluation 已退役。
**Migration**: 保留 workflow 自行定义 query leakage guard。

### Requirement: DeepSense6G Top8 selector 配置驱动工作流
**Reason**: DeepSense6G Top8 selector 已退役。
**Migration**: 无兼容迁移。

### Requirement: Top8 selector 验收命令
**Reason**: Top8 selector CLI 和配置将删除，不再有验收命令。
**Migration**: 使用保留 workflow 的 focused validation。

### Requirement: Top8 selector README 工作流说明
**Reason**: README 不再推荐 Top8 selector。
**Migration**: 无兼容迁移。
