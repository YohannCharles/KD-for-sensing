## Why

项目已经完成多处从 legacy/compat 入口到 canonical 模块的迁移，但源码、配置、测试和文档仍要求保留旧 dataset type、旧 facade、旧配置别名和原代码兼容路径。这些兼容层继续扩大维护面，并让新实验需要绕过已经过时的入口。

## What Changes

- **BREAKING** 删除可执行代码中的 legacy/compat 冗余入口，默认只支持 canonical DeepSense6G dataset、窄模块导入和 canonical 配置命名。
- **BREAKING** 将 `scenario9.py` 承载的 DeepSense6G dataset 主实现迁移到场景中立模块，并删除 `scene-specific dataset class alias`、`scene-specific dataset class alias`、`scene-specific dataset class alias` 兼容类和 `the scene-9 dataset-type spelling|scenario31|scenario32` 旧入口。
- **BREAKING** 删除 `the builder facade module`、`the transform facade module`、`the transform aggregate module` 等已由窄模块替代的兼容 facade；内部代码、测试和文档统一改用职责明确的模块路径。
- **BREAKING** 删除 legacy fusion 配置别名、旧 fusion 类名 alias、原代码兼容权重 fallback 和其它只为历史命名存在的兼容逻辑；保留仍是当前实验主路径的 canonical 配置、注册名和 checkpoint 诊断能力。
- 更新 README、扩展指南、OpenSpec 说明和测试，把“旧入口继续兼容”的要求改为“旧入口被拒绝并给出迁移提示”或直接删除旧入口说明。
- 增加架构回归检查，确保新增代码不会重新引用已删除的兼容 facade、legacy dataset type 或 legacy 配置路径。

## Capabilities

### New Capabilities
- 无

### Modified Capabilities
- `project-architecture`: 从“允许短期兼容 facade”改为“删除已迁移的二级兼容层，并禁止内部与公开推荐路径继续依赖这些入口”。
- `component-registry`: 删除场景专用 dataset alias 和旧模型类名 alias 的注册/导出要求，要求 registry 只暴露 canonical 数据集和模型入口。
- `deepsense6g-scene-selection`: 删除 `scenario9|scenario31|scenario32` 作为 dataset type 的兼容语义，保留 `data.dataset.type: deepsense6g` 加 `data.dataset.scene` 的场景选择路线。
- `modality-aware-data-loading`: 将 Scenario 9 专用 dataset 契约收敛为场景中立 DeepSense6G dataset 契约，并要求旧 dataset type 被拒绝或迁移到 canonical 配置。
- `configurable-multimodal-fusion`: 删除 legacy fusion 配置别名和旧 fusion 类名 alias 的兼容要求，要求文档和测试只使用 canonical fusion 配置。
- `original-code-compatibility`: 取消原代码兼容配置、随附 legacy 权重 fallback 和旧输入结构的运行兼容承诺，只保留严格 checkpoint 诊断和必要迁移说明。
- `experiment-artifact-registry`: 删除场景化 legacy 权重 fallback 解析，要求 teacher checkpoint 来源通过 canonical registry 或显式路径配置。
- `modality-visual-diagnostics`: 删除旧可视化兼容入口作为可运行工作流的要求，统一推荐 Gradio viewer manifest/export 路线。

## Impact

- 影响代码：`src/kd_sensing/data/datasets/scenario9.py`、`src/kd_sensing/data/datasets/__init__.py`、`src/kd_sensing/data/__init__.py`、`src/kd_sensing/data/scenes.py`、`src/kd_sensing/registries.py`、`src/kd_sensing/data/transforms.py`、`src/kd_sensing/data/transform_ops/_legacy.py`、`src/kd_sensing/engine/builders.py`、fusion 模型/配置 loader、artifact registry、诊断 CLI 和相关导入边界测试。
- 影响配置与文档：`configs/` 中 legacy 命名入口、README、`docs/extension_guide.md`、诊断/训练说明和当前 active OpenSpec 变更中仍声明兼容旧入口的内容。
- 影响测试：需要删除或重写所有直接导入 `scene-specific dataset class alias`、`the builder facade module`、`the transform facade module`、legacy fusion config alias、旧类名 alias 和 old checkpoint fallback fallback 的测试；新增旧入口拒绝与无残留引用检查。
- 影响用户：旧配置和旧 import 路径会失败，需要迁移到 `data.dataset.type: deepsense6g`、`data.dataset.scene: <scene>`、窄模块导入、canonical fusion 配置和显式 checkpoint/teacher registry 路径。
