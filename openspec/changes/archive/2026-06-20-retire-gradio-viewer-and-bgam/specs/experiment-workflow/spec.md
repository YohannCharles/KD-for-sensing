## ADDED Requirements

### Requirement: BGAM 和 viewer manifest 当前实验入口已退役
当前训练、评估、quickstart、CLI help、run metadata、配置加载和推荐文档 MUST 不再包含 BGAM、viewer manifest、仓库级 Gradio viewer 或 `kd-sensing-visualize-modalities` 入口。旧 BGAM 配置、console script、run plan、debug mask、manifest export 和 viewer prediction export 不得作为当前 workflow 兼容承诺。

#### Scenario: CLI help 不包含 BGAM 或 viewer manifest
- **WHEN** 开发者执行当前推荐的 CLI help 验证
- **THEN** 验证 MUST 不要求 `kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities` 或任何 `*-bgam*` 命令存在
- **AND** 若这些命令被调用，系统 MUST 不提供当前可运行 workflow

#### Scenario: 旧 BGAM 配置路径失败
- **WHEN** 用户传入 `configs/deepsense6g_gps_lidar_bgam.yaml`、`configs/mmw_town_gps_lidar_bgam.yaml` 或其它 BGAM 配置路径
- **THEN** 配置加载 MUST 失败或报告路径已退役
- **AND** 系统 MUST 不生成等价 virtual config

## MODIFIED Requirements

### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、`model.primary` 主模型、supervised/adaptation/JEPA/CSI 或诊断目标、训练超参数、优化器、调度器、输出目录、随机种子、GPS 特征模式和 fusion 模态选择。当前支持的训练配置 MUST 不覆盖 KD 模式、teacher checkpoint、BGAM、viewer manifest 或仓库级 Gradio viewer；旧 KD、teacher/student no-KD、Hist、Top8 standalone、residual、camera residual、BGAM 和 viewer manifest 路径 MUST 在配置解析或 registry 层被拒绝。

#### Scenario: 使用当前 JEPA 和保留 workflow
- **WHEN** 用户运行当前 JEPA pretraining/downstream、GPS-query pooling、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark 或其它保留配置
- **THEN** 配置加载 MUST 成功
- **AND** 系统 MUST 构建单一 `model.primary` 主模型或对应诊断 workflow
- **AND** 系统 MUST 不要求 BGAM、viewer manifest 或 Gradio viewer 模块存在

#### Scenario: 退役配置被拒绝
- **WHEN** 用户传入旧 KD、Hist、Top8 standalone、residual、camera residual、BGAM 或 viewer manifest 专属配置
- **THEN** 配置解析 MUST 失败并给出清晰迁移或退役提示
- **AND** 错误信息 MUST 指向当前 `model.primary`、supervised/adaptation、JEPA、CSI 或保留 baseline 入口

### Requirement: 默认实验入口去 KD-first 化
项目默认 quickstart、README 推荐入口、当前主线 quick validation 和新 canonical mainline 配置 MUST 以 supervised/adaptation、JEPA、CSI hardening、baseline/control、保留诊断或 benchmark 工作流为默认。旧 KD、BGAM 和 viewer manifest 配置不得作为当前主线默认实验入口。

#### Scenario: README quickstart 使用当前主线
- **WHEN** 开发者阅读 README 或当前主线运行说明
- **THEN** 推荐的首个训练、评估或诊断命令 MUST 使用当前 supervised/adaptation、JEPA、CSI、baseline/control 或保留诊断配置
- **AND** 文档 MUST 不把 `logits_kd`、`rkd`、Hist/HiST、standalone Top8 selector、GPS residual、camera residual、BGAM 或 viewer manifest 作为当前主线 quickstart

#### Scenario: canonical mainline 配置不要求 teacher checkpoint
- **WHEN** 用户加载当前推荐的 mainline 配置
- **THEN** 配置 MUST 能在没有 teacher checkpoint 的情况下完成解析和 dry-run/smoke 构建
- **AND** 输出 metadata MUST 不记录 KD-enabled、BGAM-enabled 或 viewer-manifest lineage

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态 beam prediction、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、GPS v2/adapter、MMW Town GPS v2、CSI hardening、JEPA visual analysis、预处理和保留诊断，而不是 KD-first、HiST-Beam-first、Raymobtime-first、Top8/residual-first、GPS coarse-anchor-first、BGAM-first 或 viewer-first 工作流。历史 KD、Hist、Raymobtime、Top8 selector、residual、camera residual、GPS coarse anchor、BGAM 或 Gradio/viewer 背景可以保留在 archive 或历史说明中，但必须标记为已退役或历史记录。

#### Scenario: pyproject 描述不再 KD Hist 或退役路线 first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation、HiST-Beam、Top8 selector、residual、GPS coarse anchor、BGAM 或 viewer manifest 描述为当前唯一或首要工作流
- **AND** 若提到这些路线，MUST 表达其为 legacy、historical 或 retired

#### Scenario: 文档保留历史说明
- **WHEN** README 或 docs 提到历史 KD、Hist、Top8 selector、residual、camera residual、GPS coarse anchor、BGAM 或 Gradio viewer 代码
- **THEN** 文档 MUST 说明对应能力已从当前 active mainline 退役
- **AND** 文档 MUST 不提供当前推荐运行命令

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理、保留诊断和通用训练评估。KD baseline、HiST-Beam/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS coarse anchor、residual fusion、camera residual、BGAM、viewer manifest、仓库级 Gradio viewer、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、supporting、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、Raymobtime s008、retired Top8 selector/residual/GPS coarse anchor 命令、BGAM 命令、viewer manifest 命令或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、Top8 selector、residual、camera residual、GPS coarse anchor、BGAM、viewer manifest、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/JEPA/CSI 主线

#### Scenario: 当前 workflow 文档声明运行状态
- **WHEN** 文档列出当前实验配置、benchmark manifest 或诊断配置
- **THEN** 文档 MUST 标明该条目是 formal、lowmem、smoke、debug、evaluation-only、upper-bound、historical ablation 还是 mock
- **AND** upper-bound、mock、smoke 或 historical ablation MUST 不得被写成正式结论

### Requirement: 健康检查反映保留入口
快速健康检查 MUST 覆盖当前仍支持的架构边界、包内 CLI、JEPA visual analysis、文档健康和当前主线 focused tests。健康检查 MUST 不要求 Raymobtime s008、BGAM、viewer manifest、已退役的模态失衡诊断脚本、fusion KD virtual alias 或 HiST-Beam/Hist CLI 可用。

#### Scenario: focused validation 不依赖退役入口
- **WHEN** 开发者执行本 change 的 focused 验证
- **THEN** 验证命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 不包含已退役的 Hist CLI、Hist configs、BGAM CLI/config/tests、viewer manifest CLI 或独立模态诊断脚本
- **AND** 验证 MUST 覆盖配置加载失败、架构边界、registry 和保留 evaluation subset 能力

### Requirement: 现有 supervised/adaptation workflow 不变
新增或删除非保留 workflow MUST 不改变现有 beam、occlusion、position、multitask、GPS v2、CSI hardening、JEPA 或 supervised fusion workflow 的默认配置和指标。Raymobtime s008、legacy KD、standalone Top8 selector、residual、BGAM 和 viewer manifest 路线只作为退役或历史 guard 语义保留，不属于当前默认 workflow。

#### Scenario: 保留 workflow 配置不引入退役依赖
- **WHEN** 用户加载当前 supervised/adaptation、JEPA、GPS v2 或 CSI 配置
- **THEN** 配置 MUST 不要求 BGAM、viewer manifest、Raymobtime s008、legacy KD、standalone Top8 selector 或 residual 模块存在
- **AND** 默认指标和输出目录 MUST 保持该保留 workflow 自身语义

## REMOVED Requirements

### Requirement: GPS+LiDAR BGAM 配置驱动工作流
**Reason**: BGAM 配置驱动 workflow 已退役。
**Migration**: 无兼容迁移；删除 BGAM 配置、CLI、产物要求和 README 命令。

### Requirement: GPS+LiDAR BGAM 验收命令
**Reason**: BGAM CLI、配置和测试将删除，不再有验收命令。
**Migration**: 使用保留 workflow 的 focused validation 和最终回归。

### Requirement: GPS+LiDAR BGAM README 工作流说明
**Reason**: README 不再推荐或说明 BGAM 当前运行方式。
**Migration**: 历史说明如保留，必须标记为 retired/historical 且不提供当前命令。
