## Why

项目已经完成多轮入口、配置和模块边界收敛，但仍有若干高变更频率模块承担过多职责，例如 objective metadata、Multimodal-NF common helper、viewer manifest、DeepVerse label builder 和部分诊断 writer。继续在这些文件中叠加功能会增加回归风险，也会让新实验能力难以定位维护边界。

本变更制定热点模块拆分方案和验收边界，以小步、兼容、测试驱动的方式降低单文件复杂度，不改变用户可见训练、评估、预处理和诊断语义。

## What Changes

- 定义热点模块拆分原则：先移动纯 helper/schema/table/writer，再移动业务分支；公开入口和 import 语义保持兼容。
- 为 `objective_metadata` 拆分 objective registry、metric aliases、history/TensorBoard schema 和 validation helper。
- 为 Multimodal-NF preprocessing/common helper 拆分 path resolution、audit、index rows、split assignment、codebook metadata 和 HDF5 inspection。
- 为 viewer manifest/diagnostics 继续拆分 prediction merge、cache metadata、sample record schema、model prediction export 和 manifest writer。
- 为 DeepVerse label builder 拆分 scene metadata、target derivation、split assignment、sanity checks 和 output writers。
- 增加架构边界测试，防止拆分后出现新的二级兼容聚合层或重依赖导入回流。
- 明确非目标：不删除历史公开入口，不重命名用户配置，不改变 metrics 字段和 artifact 文件名。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `project-architecture`: 增加热点模块拆分、兼容 facade 边界和重依赖导入回归防护要求。
- `experiment-workflow`: 锁定拆分后训练/评估 artifact 字段兼容性。
- `multimodal-nf-dataset`: 约束 Multimodal-NF 预处理/index/helper 拆分后契约保持不变。
- `gradio-visual-analysis`: 约束 viewer manifest 相关拆分后公开入口和 manifest schema 保持兼容。

## Impact

- 主要影响源码组织和测试，不应改变用户命令、配置路径、输出目录或已有 artifact schema。
- 可能新增若干窄模块，例如 `engine/objectives/`、`preprocessing/multimodal_nf_*`、`diagnostics/viewer_manifest_*`、`data/deepverse/*`。
- 测试重点是导入边界、旧公开入口兼容、配置加载、focused unit tests 和少量 artifact snapshot 断言。
