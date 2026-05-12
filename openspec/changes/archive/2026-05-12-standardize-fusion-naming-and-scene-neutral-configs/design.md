## Context

项目已经有 canonical fusion 配置生成器、legacy fusion 示例配置、CRAF/MARF/G2D 诊断配置和多场景 viewer 工作流。当前问题不是训练语义错误，而是命名边界不清：

- `fusion_student` 的实现类叫 `StudentModalityNet`，与 image/GPS/LiDAR/mmWave 的 `*StudentModalityNet` 风格不一致，也难以通过代码搜索判断它只属于 fusion。
- `fusion_teacher` 的实现类叫 `FusionModalityNet`，没有直接表达 teacher 角色。
- 一批 CRAF/MARF 配置使用 `scene32_` 文件名、`experiment.name` 和 `output.run_name`，把数据集场景混进实验方法名。
- `engine._builders_impl`、`data.transform_ops._legacy` 和旧 viewer manifest 脚本已经主要是转发层，容易让新代码继续引用过时路径。

本变更需要同时处理 OpenSpec 约束、代码命名、配置文件名、文档和测试，避免只做局部改名后留下新的兼容歧义。

## Goals / Non-Goals

**Goals:**

- 为 legacy early-concat fusion 模型提供职责明确的新公开类名：`FusionTeacherModalityNet` 和 `FusionStudentModalityNet`。
- 保留旧类名 `FusionModalityNet` 和 `StudentModalityNet` 作为短期兼容 alias，避免立即破坏已有 notebook、测试和 checkpoint 加载。
- 去掉推荐配置中的 `scene32_` 前缀；场景选择保留在 dataset 配置、输出根目录和运行参数中。
- 更新测试和文档，使新代码路径默认使用新类名和场景中立配置名。
- 将二级兼容聚合层和重复脚本列入退役任务，优先把内部调用点切到窄模块或包内 CLI。

**Non-Goals:**

- 不改变 fusion teacher/student 的网络结构、参数名、state_dict key 或 checkpoint 加载语义。
- 不删除 `All_models` 内置复现权重，不改变 image/image+radar 原代码兼容配置要求。
- 不改变 Scene 32 默认数据集选择；只去掉命名前缀，不改变 `scene_id: 32` 或 `scene_slug: scene32` 的数据语义。
- 不在本批次全面移除所有旧 facade；公开兼容入口需要先有迁移窗口。

## Decisions

### 1. 用 subclass 或直接重命名承载新 fusion 类名，旧名作为 alias

首选实现是把实际类名改为 `FusionTeacherModalityNet` / `FusionStudentModalityNet`，并在同模块末尾提供：

- `FusionModalityNet = FusionTeacherModalityNet`
- `StudentModalityNet = FusionStudentModalityNet`

注册名继续保持 `fusion_teacher` / `fusion_student`，因为配置注册名已经是清晰的公共 API。这样 state_dict key 不会因类名变化而改变，也不会影响 checkpoint。

替代方案是新增 subclass 包装旧类，但 `type(model).__name__` 仍可能根据构造路径分裂；直接让注册表指向新类名更清晰。

### 2. 配置文件名和 run name 去掉 `scene32_`，dataset 字段保留场景

`scene32_marf.yaml` 这类文件改为 `marf.yaml`，`experiment.name` / `output.run_name` 同步改为 `marf`。相关 ablation 配置同样去掉 `scene32_` 前缀。

场景信息仍通过 `data.dataset.scene`、`scene_id`、`scene_slug`、`outputs/scene32/...` 和命令行覆盖表达。这样同一方法配置可以被 Scene 9/32 复用，运行目录仍可由输出根目录或 registry metadata 区分场景。

替代方案是只改 README 示例，但保留文件名不动；这会让新实验继续复制 `scene32_` 模式。

### 3. 兼容层分层退役

保留公开 facade：

- `kd_sensing.engine.builders`
- `kd_sensing.data.transforms`
- `kd_sensing.diagnostics.modality_visualization`

优先退役或停止内部引用二级兼容层：

- `kd_sensing.engine._builders_impl`
- `kd_sensing.data.transform_ops._legacy`
- `tools/visualization/export_viewer_manifest.py` 中与包内 CLI 重复的 parser/main

替代方案是一次性删除所有兼容层，但当前 tests 和文档仍覆盖旧 import 兼容；更稳妥的做法是先把新代码迁出，再在后续 breaking change 删除公开 facade。

## Risks / Trade-offs

- 旧类名 alias 继续存在会让命名不一致短期内仍可被搜索到 → 文档和测试全部改用新类名，alias 只服务旧 import。
- 重命名配置文件可能让用户旧命令失效 → 在 README 中给出旧名到新名映射；如需要可保留极小 YAML shim 或明确迁移命令。
- `scene32_` 去掉后输出目录方法名变短，历史输出和新输出不再同名 → 这是预期变化，场景信息应由输出根目录或 metadata 表达。
- 删除 `_legacy` / `_builders_impl` 如果过早可能破坏外部私有引用 → 本批次先验证内部引用和测试，不把私有路径当长期 API。

## Migration Plan

1. 新增 fusion 新类名并更新导出；旧类名作为 alias 保留。
2. 批量更新 tests、docs、README 和内部 import，优先使用新类名。
3. 将 `configs/fusion/scene32_*.yaml` 重命名为场景中立文件，并同步内部 `experiment.name` / `output.run_name`。
4. 更新引用这些配置的 README、tools 文档、analysis 配置和测试。
5. 把内部脚本从 `engine.builders` 或重复 tools manifest exporter 切到窄模块 / 包内 CLI。
6. 使用 `conda run -n kd_mm_beam pytest ...` 跑 fusion、MARF/CRAF、配置矩阵和架构边界相关测试。

## Open Questions

- 旧 `configs/fusion/scene32_*.yaml` 是否需要保留为 shim 文件一个版本周期，还是直接删除并让缺失路径暴露迁移错误？
- `kd-sensing-visualize-modalities` 是否在本批次继续保留，还是后续单独做 breaking change 移除？
