## ADDED Requirements

### Requirement: BGAM 和 viewer manifest 退役防回流
项目 MUST 从当前源码支持面删除 BGAM 与 viewer manifest/Gradio viewer 相关公开入口、配置、专属实现模块、测试和推荐文档。系统 MUST 不新增兼容 alias、stub CLI、薄 facade、虚拟配置、registry fallback 或旧 viewer manifest helper 来维持这些退役路线可发现性。

#### Scenario: 退役入口不可安装
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** `pyproject.toml` MUST 不声明 BGAM、viewer manifest 或 `kd-sensing-visualize-modalities` 相关命令
- **AND** 安装生成的 entry points MUST 不包含 `kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities`、`kd-sensing-run-*-bgam`、`kd-sensing-evaluate-*-bgam` 或 `kd-sensing-prepare-*-bgam-manifest`

#### Scenario: 退役实现不可作为当前模块导入
- **WHEN** 开发者检查 `src/kd_sensing/cli`、`data`、`engine`、`models`、`losses` 和 `diagnostics`
- **THEN** BGAM 专属模块和 `viewer_manifest*` 专属模块 MUST 不再作为当前源码模块保留
- **AND** 保留主线不得从这些退役模块导入 helper

## MODIFIED Requirements

### Requirement: 诊断可视化内部模块化
诊断可视化入口 MUST 收敛为仍保留的 JEPA visual analysis、GPS shortcut benchmark 和其它明确 current 的包内诊断能力。项目 MUST 不再维护旧静态 modality visualization PNG workflow、`kd_sensing.diagnostics.visualization` 内部渲染模块、仓库级 Gradio viewer support、viewer manifest 导出或 `kd-sensing-visualize-modalities` 兼容 alias。

#### Scenario: viewer manifest 导出入口已退役
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 不存在或失败并说明 viewer manifest 已退役
- **AND** 项目 MUST 不通过其它 console script 恢复 viewer manifest 导出

#### Scenario: modality visualization 兼容 alias 已退役
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 不存在或失败并说明旧 modality visualization/viewer alias 已退役
- **AND** 该入口 MUST 不再委托任何 viewer manifest、旧 PNG 总览图或仓库级 Web UI

#### Scenario: JEPA visual analysis 作为保留诊断出口
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`
- **THEN** 命令 MUST 正常退出
- **AND** 该入口 MUST 使用 `kd_sensing.diagnostics.jepa_visual_analysis` 生成本地分析 manifest、图表、表格和 report

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的兼容 console script MUST 是薄 alias，不得复制长期维护的 parser 或主实现。项目 MUST 不再要求安装 `kd-sensing-raymobtime-analysis`、GPS window baseline、仓库级 Gradio viewer support、viewer manifest 或 BGAM 入口。BeamBench 相关 console scripts MAY 保持当前声明。

#### Scenario: 已退役可视化入口不可用
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** 安装生成的 entry points MUST 不包含 `kd-sensing-export-viewer-manifest` 或 `kd-sensing-visualize-modalities`
- **AND** README MUST 不把这些命令列为当前安装后可用入口

#### Scenario: 已退役 BGAM 入口不可用
- **WHEN** 开发者刷新 editable install 后检查 console scripts
- **THEN** 安装生成的 entry points MUST 不包含 DeepSense6G/MMW BGAM prepare、run 或 evaluate 命令
- **AND** 项目 MUST 不提供等价 `python -m kd_sensing.cli.*bgam*` 当前入口

#### Scenario: 保留入口仍一致
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含当前保留的训练、评估、预处理、run-index、cleanup、JEPA visual analysis、JEPA benchmark 和其它 current workflow 命令
- **AND** 安装生成的 entry points MUST 不要求包含 BGAM、viewer manifest、Raymobtime analysis、GPS window baseline 或仓库级 Gradio viewer support 入口

### Requirement: Active mainline 与 legacy KD 模块边界
项目 MUST 区分当前主线方法模块、supporting helper 和 legacy/retired 模块。当前主线包括 supervised beam prediction、Image+GPS JEPA query-pool downstream、paired baseline/control、Vision-Position baseline suite、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、soft-label supervised training 和通用训练/评估能力。DeepSense6G/MMW BGAM、viewer manifest、HiST/Hist、GPS residual、camera residual、standalone Top8 selector、Raymobtime s008、CRAF/MARF/G2D、Multimodal-NF 和旧 KD MUST 不作为 active mainline 描述；若仍有通用 helper 被保留，MUST 标记为 supporting 或迁移边界。

#### Scenario: 导入当前主线不触发退役模块
- **WHEN** 开发者导入当前主线的训练、评估、JEPA downstream、CSI hardening、soft-label helper 或保留诊断模块
- **THEN** 系统 MUST 不导入 BGAM、viewer manifest、旧 KD、Hist、Raymobtime s008、G2D、CRAF、MARF 或 Multimodal-NF 专属模块
- **AND** 退役模块缺失 MUST 不影响当前主线导入

#### Scenario: inventory lifecycle 与主线一致
- **WHEN** 开发者查看项目 surface inventory
- **THEN** BGAM 与 viewer manifest MUST 标记为 retired-tombstone 或从 current 支持面移除
- **AND** inventory MUST 不把它们列为当前热点、owner facade 或 validation requirement

### Requirement: Top8 residual coarse 退役边界
Top8 selector 训练/plot/compare、GPS coarse anchor、GPS prior residual/delta correction、camera residual、BGAM、BGAM 依赖的 TopK candidate manifest/dataset/loss 支撑代码、仓库级 Gradio viewer 和 viewer manifest MUST 不属于当前包结构和推荐入口。通用 Top-K 指标、circular metrics、GPS-Rel-Polar、GPS v2、CSI 和 JEPA MAY 保留；Raymobtime 旧名称只允许作为 migration guard 或退役说明出现。

#### Scenario: 保留通用指标
- **WHEN** 清理实现扫描到 `topk`、`candidate`、`viewer` 或 `bgam` 字符串
- **THEN** 系统 MUST 按语义判断归属
- **AND** 普通 evaluation Top-K、CSI candidate ranking 和 GPS v2 自身诊断不得仅因字符串命中被删除

## REMOVED Requirements

### Requirement: Viewer manifest 实现不得集中在聚合模块
**Reason**: viewer manifest 导出和 helper 已退役，不再维护聚合模块拆分契约。
**Migration**: 无兼容迁移；若通用 helper 保留，必须迁入非 viewer 命名模块。

### Requirement: Viewer manifest 轻量 helper import 边界
**Reason**: `viewer_manifest_*` helper 不再作为 current 轻量导入边界。
**Migration**: 保留诊断由各自 import boundary tests 约束。

### Requirement: GPS+LiDAR BGAM 包内入口
**Reason**: GPS+LiDAR BGAM workflow 已退役，不再需要包内入口、CLI、模型、loss 或 eager import 边界。
**Migration**: 无兼容迁移。
