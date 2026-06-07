## Why

当前项目已经转向 DeepSense6G/MMW/Raymobtime 的 supervised、adapter、GPS candidate、residual fusion 和诊断主线，但源码、配置、README、OpenSpec 和本地 `outputs/` 仍保留大量 HiST-Beam/HiST 研究线入口与历史实验产物。继续维护这些表面会放大测试矩阵、混淆推荐工作流，并让 `src/kd_sensing/engine` 与 `src/kd_sensing/models` 中已经不用的 Hist 模型代码长期占据主路径。

## What Changes

- **BREAKING** 退役 HiST-Beam/Hist 模型工作流：删除 `kd-sensing-hist-beam-loso`、`configs/hist_beam/`、`src/kd_sensing/engine/hist_beam_*`、`src/kd_sensing/models/fusion/hist_beam.py`、Hist 专用 evaluation/helpers、相关 tests、docs 和 registry 入口。
- **BREAKING** 组件注册表不再暴露 `hist_beam_fusion`、HiST-Beam variants、path/radio prototype、history-anchor Hist residual 或 image-only HiST probe 作为可构建模型/工作流。
- **BREAKING** README、docs、pyproject 和 OpenSpec 当前规范不再声明 HiST-Beam、HiST LOSO、P3-HiST、radio-semantic HiST 或 MMW sensor-assisted HiST 为支持入口；历史 archive 只作为旧记录保留。
- 整理 `src/kd_sensing/engine` 和 `src/kd_sensing/models` 的文件边界：保留当前主线 engine/model 文件，删除 Hist 专用模块，避免新增兼容 facade 或旧入口 alias。
- 整理 `outputs/`：先生成可审计 runtime cleanup manifest，再删除明确过时的 Hist、P3/V8/V9 probe、debug/plan-check、smoke、stale/failed 和语义不清历史产物；不得删除 `dataset/`、`All_models/`、源码、配置、OpenSpec、已跟踪文件或未匹配的活跃实验。
- 将后续默认输出结构收敛为按用途分组的目录：训练 run、analysis artifact、cache、features、cleanup manifest 和场景级 best checkpoint 分开管理，避免新的实验继续落入 `outputs/other/` 或临时根目录。

## Capabilities

### New Capabilities

- `project-surface-cleanup`: 定义退役研究线、源码表面整理和输出目录分区的全局清理契约。

### Modified Capabilities

- `project-architecture`: 包结构和健康检查不再要求 Hist/HiST-Beam engine、model、CLI、config 或 evaluation 模块存在，并要求退役研究线不得留下旧入口兼容层。
- `component-registry`: 默认注册组件不再包含 Hist/HiST-Beam 模型和变体；旧 Hist 模型名必须 fail fast。
- `experiment-workflow`: 推荐训练、评估、quickstart、CLI help、run metadata 和文档工作流不再包含 HiST-Beam/Hist LOSO 入口。
- `runtime-artifact-cleanup`: 清理规则扩展为可识别已退役 Hist/P3/V8/V9/debug/smoke/plan-check 输出，并支持在 manifest 审计后删除这些本地产物。
- `experiment-artifact-registry`: checkpoint 保留策略需要保护当前主线复现必需 artifact，同时允许退役 Hist 产物进入候选。
- `hist-beam-cross-scene-adaptation`: 从支持能力改为退役能力；源码不再提供 HiST-Beam 训练、适配、评估或输出契约。
- `cross-scene-loso-workflow`: LOSO 工作流不再绑定 HiST-Beam 默认矩阵；若未来需要跨场景矩阵必须由当前主线 workflow 重新定义。
- `history-anchored-residual-beam`: 退役依赖 Hist 模型实现的 history-anchor residual 路径；保留独立 GPS/window baseline 或其它当前主线能力不受影响。
- `path-prototype-hist-beam-adaptation`: 退役 P3/HiST-Beam path prototype 能力和相关输出契约。
- `radio-semantic-hist-beam-adaptation`: 退役 radio-semantic HiST-Beam 能力和相关 prototype/evaluation 契约。
- `image-only-legal-crossroad-probe`: 退役 image-only HiST probe 入口和相关输出契约。
- `mmw-sensor-assisted-beam-prediction`: 删除对 sensor-assisted HiST-Beam profile 的要求，MMW 当前主线继续由非 Hist workflow 承担。

## Impact

- 源码：`src/kd_sensing/cli/hist_beam_loso.py`、`src/kd_sensing/engine/hist_beam_*`、`src/kd_sensing/models/fusion/hist_beam.py`、Hist 专用 evaluation 模块、registry/default imports、training extension 挂钩和 profiling helper。
- 配置：删除 `configs/hist_beam/` 及引用 HiST-Beam variant/profile 的配置；旧路径不新增 virtual alias。
- 测试：删除或改写 `test_hist_beam_*`、history-anchor Hist、image-only Hist、V7/V8/V9 Hist、CLI help 和训练 IO 中的 Hist 断言；新增旧入口拒绝和输出清理 manifest 测试。
- 文档/OpenSpec：README、docs、pyproject 和相关 specs 从“支持 HiST-Beam”改为“Hist 研究线已退役；历史 archive 仅只读保留”。
- 本地产物：`outputs/` 中过时实验会通过 manifest 进入候选并在显式删除阶段移除；当前主线分析、cache、features、best checkpoints 和未匹配运行默认保留。
