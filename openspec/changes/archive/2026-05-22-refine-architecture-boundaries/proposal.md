## Why

当前代码架构整体已经按配置、注册表、模态契约、engine runtime 和训练扩展点形成清晰边界，但架构体检发现仍有三类沉积：OpenSpec/文档健康检查存在误伤与 purpose 过短问题，配置加载会因 objective 元数据导入 torch 运行时实现，诊断可视化子模块虽然职责已拆分但部分轻量模块仍导入渲染和数据构建重依赖。此变更用于收紧这些边界，避免继续扩大抽象层或重构主训练流程。

## What Changes

- 修正 OpenSpec 文档健康检查语义，避免检测规则因 spec 正文描述待拒绝字符串而误伤，同时补齐过短或遗留的 capability purpose。
- 将 prediction objective 的纯元数据、默认 metric、early stopping alias、history/TensorBoard 字段等轻量契约与 torch loss 计算职责分离，使 `kd_sensing.config` 不因 objective 默认补全导入 torch。
- 收紧诊断可视化内部 import 边界，使 `diagnostics.visualization.config`、`sampling`、`writers` 等非渲染/非数据构建模块不导入 matplotlib、PIL、torch、dataset builder 或 pandas 等不必要重依赖。
- 增加 focused 架构边界测试，覆盖 config 轻量导入不导入 torch、可视化轻量子模块不导入渲染栈、OpenSpec purpose 健康检查不再自引用误伤。
- 不改变训练、评估、预处理、viewer manifest 的公开 CLI 行为，不新增旧入口或兼容聚合层。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 收紧轻量导入、OpenSpec 文档健康检查和诊断可视化内部 import 边界要求。
- `first-class-prediction-tasks`: 明确 prediction objective 元数据是轻量契约，必须可被配置加载复用且不导入 torch loss/runtime 实现。
- `modality-visual-diagnostics`: 明确 manifest/viewer 诊断内部的配置、采样、写出 helper 与渲染、数据集构建、模型预测等重依赖职责分离。

## Impact

- 受影响源码主要包括 `src/kd_sensing/engine/prediction_objectives.py`、可能新增的 objective metadata/helper 模块、`src/kd_sensing/config/normalization.py`、`src/kd_sensing/diagnostics/visualization/*` 和架构边界测试。
- 受影响 OpenSpec artifacts 包括 `project-architecture`、`first-class-prediction-tasks`、`modality-visual-diagnostics` 的 delta specs，以及当前 change 的 proposal、design 和 tasks。
- 不引入新第三方依赖，不改变配置文件语义、checkpoint 格式、训练输出目录结构或 console script 名称。
