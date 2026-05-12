## Why

当前 fusion 代码和实验配置里同时存在泛化能力命名与具体场景命名：`StudentModalityNet` 过于宽泛，`scene32_*` 配置名把场景编号固化进模型实验身份。随着 Scene 9/32 共用训练、诊断和可视化链路，这会让模型类型、配置 slug、checkpoint registry 与文档检索产生歧义。

这次变更把 fusion 模型命名、配置命名和兼容边界收紧：模型类名表达职责，实验配置名保持场景中立，场景编号只留在数据集字段、输出目录或显式运行参数里。

## What Changes

- 将 legacy early-concat fusion student 的公开实现名标准化为 `FusionStudentModalityNet`，保留 `StudentModalityNet` 作为短期兼容 alias。
- 将 legacy early-concat fusion teacher 的公开实现名补齐为 `FusionTeacherModalityNet`，保留 `FusionModalityNet` 作为短期兼容 alias。
- 更新模型包导出、测试、文档和类型断言，使新代码优先使用 `FusionTeacherModalityNet` / `FusionStudentModalityNet`。
- 将当前 `configs/fusion/scene32_*.yaml` 改为场景中立命名；配置内部的 `experiment.name`、`output.run_name` 和 README 示例不再出现 `scene32_` 前缀。
- 保留数据集 `scene` / `scene_id` / `scene_slug`、checkpoint registry 的场景目录和运行时输出目录语义；本变更不改变训练数据选择。
- 梳理冗余兼容层：`engine._builders_impl`、`data.transform_ops._legacy`、旧 viewer manifest 脚本和旧 visualize entry point 进入明确退役计划，但本批次只移除低风险的内部重复或把内部调用点切到窄模块。
- 不删除原论文 image/image+radar 兼容权重和 legacy config 语义；这些仍由现有 `original-code-compatibility` 要求保护。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `configurable-multimodal-fusion`: fusion 模型公开类名必须区分 teacher/student 职责，并保留旧类名 alias 作为兼容入口。
- `experiment-workflow`: 实验配置 slug 和 run name 不得硬编码 `scene32_` 前缀；场景选择必须由 dataset/CLI/output 语义表达。
- `project-architecture`: 内部代码应优先使用窄模块和新 fusion 类名；只保留必要公开 facade，私有二级兼容聚合层进入退役边界。

## Impact

- 影响代码：`src/kd_sensing/models/fusion/`、`src/kd_sensing/models/__init__.py`、相关训练/诊断配置、README、扩展文档和测试。
- 影响配置：`configs/fusion/scene32_*.yaml` 会改为不含场景前缀的文件名和 run name；旧路径如需保留，应只作为兼容别名或文档迁移提示，不再作为推荐入口。
- 影响 checkpoint/输出：新 run name 不再带 `scene32_`，但输出目录仍可通过 `outputs/scene32/...` 或数据集场景元数据表达场景。
- 影响兼容性：旧 `FusionModalityNet` / `StudentModalityNet` import 暂不破坏；新代码、文档和测试改用更明确的新名称。
