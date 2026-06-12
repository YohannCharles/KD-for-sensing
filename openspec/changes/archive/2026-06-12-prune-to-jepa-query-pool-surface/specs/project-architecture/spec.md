## ADDED Requirements

### Requirement: 当前支持面收敛到 Image+GPS JEPA query-pool
项目 MUST 将当前推荐训练、评估、诊断和实验配置支持面收敛到 Image+GPS JEPA query-pool 主线及其必要对照。保留面 MUST 包含 `jepa_context_image + GPSQueryPool` JEPA downstream、`fair_gps_biased` paired baseline、supervised/random-best 控制组、vision-position baseline suite 和 `jepa_visual_analysis` 论文图/诊断出口。退役路线 MUST 不再作为 README 推荐入口、pyproject console script、架构 allowlist 或当前配置矩阵出现。

#### Scenario: README 展示当前主线
- **WHEN** 开发者阅读 README 的项目定位、主要入口和配置矩阵
- **THEN** 文档 MUST 把 Image+GPS JEPA query-pool、paired baseline/control、vision-position baseline suite 和 JEPA visual analysis 描述为当前主线
- **AND** 文档 MUST 不把 GPS window、DeepVerse/DT31、旧静态 modality visualization 或仓库级 Gradio viewer support 描述为当前入口
- **AND** 文档 MAY 继续保留 BeamBench/Arnold22 Camera AE+GPS Direct 当前入口和复现辅助说明

#### Scenario: 架构测试拒绝退役入口回流
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 拒绝退役的 viewer support、GPS window baseline、DeepVerse/DT31 workflow、Top8 selector dataset 和旧静态 modality visualization 文件重新出现在当前 allowlist 中
- **AND** 测试 MUST 继续允许 JEPA query-pool、paired control、vision-position baseline、BeamBench/Arnold22 Camera AE+GPS Direct 和 JEPA visual analysis 相关入口

#### Scenario: 配置矩阵只保留必要 JEPA 对照
- **WHEN** 开发者查看 `configs/fusion/experiments/jepa_image_gps/` 和实验矩阵文档
- **THEN** 当前配置 MUST 保留 query-pool、GPS-biased baseline、supervised baseline 和 random-best 控制组
- **AND** scene31-only、非 BeamBench 的 last-checkpoint 和 next-beam downstream ablation 配置 MUST 不再作为当前配置文件维护
- **AND** `beambench_fair` 相关配置 MAY 继续保留用于 Arnold22/BeamBench 口径对照

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

## MODIFIED Requirements

### Requirement: 诊断可视化内部模块化
诊断可视化入口 MUST 收敛为包内 viewer manifest 导出和 JEPA visual analysis。项目 MUST 不再维护旧静态 modality visualization PNG workflow、`kd_sensing.diagnostics.visualization` 内部渲染模块或仓库级 Gradio viewer support。`kd-sensing-visualize-modalities` MAY 作为兼容薄 alias 保留，但 MUST 委托 `kd-sensing-export-viewer-manifest`，不得恢复独立 parser、旧 PNG 总览图或 `tools/visualization/` support 依赖。

#### Scenario: viewer manifest 导出入口兼容
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: modality visualization 兼容 alias 不恢复 PNG workflow
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest
- **AND** 该入口 MUST 不导入 `kd_sensing.diagnostics.visualization.core` 或仓库级 `tools/visualization` helper

#### Scenario: JEPA visual analysis 作为论文图出口
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`
- **THEN** 命令 MUST 正常退出
- **AND** 该入口 MUST 使用 `kd_sensing.diagnostics.jepa_visual_analysis` 生成本地分析 manifest、图表、表格和 report

### Requirement: 安装入口与 pyproject 声明一致
项目 MUST 确保 editable install 后的 console scripts 与 `pyproject.toml` 的 `[project.scripts]` 声明一致。README 或工具文档中推荐的包内 CLI MUST 可在 `kd_mm_beam` 环境中直接调用。保留的兼容 console script MUST 是薄 alias，不得复制长期维护的 parser 或主实现。项目 MUST 不再要求安装 `kd-sensing-raymobtime-analysis`、GPS window baseline 或仓库级 Gradio viewer support 入口。BeamBench 相关 console scripts MAY 保持当前声明。

#### Scenario: 可视化 manifest 导出入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 manifest 导出参数，例如 `--config`、`--cache-dir`、`--scenes` 和 `--run-models`

#### Scenario: 可视化兼容入口可用
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 明确该入口导出 viewer manifest
- **AND** 该入口 MUST 委托 manifest 导出 CLI，不得复制独立 parser、旧静态 PNG 主流程或仓库级 Gradio viewer support

#### Scenario: 安装元数据刷新后入口齐全
- **WHEN** 开发者在 `kd_mm_beam` 中执行 `python -m pip install -e .`
- **THEN** 安装生成的 entry points MUST 包含 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-visualize-modalities`、`kd-sensing-export-viewer-manifest` 和 `kd-sensing-jepa-visual-analysis`
- **AND** 安装生成的 entry points MUST 不要求包含 `kd-sensing-raymobtime-analysis`、`kd-sensing-gps-window-baseline` 或仓库级 Gradio viewer support 入口
