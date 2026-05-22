## Context

当前项目已经完成从旧脚本入口到 `src/kd_sensing` 包结构、配置驱动运行、组件注册表、模态契约和 engine 扩展点的迁移。架构边界测试大部分已覆盖训练方法扩展、builder 拆分、transform 拆分、lazy package export 和源码表面积，但仍暴露出三个问题：

- OpenSpec 文档健康检查直接扫描待拒绝字符串，导致 `project-architecture` 自己描述该规则时被误判。
- `kd_sensing.config.normalization` 需要 objective 默认 metric 和 target/head 元数据，却通过 `engine.prediction_objectives` 导入了 torch loss/runtime 实现。
- `diagnostics.visualization` 的若干子模块复制了同一组重依赖 import，导致纯配置、采样或写出 helper 也导入 matplotlib、PIL、torch、dataset builder 等运行依赖。

这些问题不需要改变训练生命周期、registry 机制、模态契约或 CLI 表面。变更应保持窄修复，避免把架构继续拆成过多抽象层。

## Goals / Non-Goals

**Goals:**

- 让配置加载路径复用 objective 元数据时保持轻量，不导入 torch、dataset、model、diagnostics 或渲染栈。
- 让诊断可视化内部模块的 import 与职责一致：配置解析、采样、写出 helper 不导入渲染和数据构建重依赖。
- 修正 OpenSpec 文档健康检查，使其检查 capability purpose，而不是误伤描述该检查规则的正文。
- 补齐 OpenSpec purpose 过短或遗留文本，确保 `openspec validate --all --strict` 中此类 warning 可被逐步清理。
- 增加 focused 架构边界测试，防止上述边界回退。

**Non-Goals:**

- 不重写 `engine.trainer.train` 主生命周期。
- 不改变现有配置字段、objective 名称、训练指标、checkpoint payload 或运行产物目录结构。
- 不新增旧脚本入口、兼容 facade、二级聚合层或长期维护的 fallback wrapper。
- 不把 viewer manifest 的数据语义改成新的格式；本变更只整理内部边界。

## Decisions

### Decision 1: 将 objective 元数据拆成轻量模块

新增或等价整理一个不导入 torch 的 objective metadata 模块，负责 `PREDICTION_OBJECTIVES`、默认主指标、metric alias、metric mode、available metrics、history fields、TensorBoard scalar 映射、required targets/outputs 和 runtime metadata。`config.normalization`、config validation、training state、training metrics 等只需要元数据的路径改为导入该轻量模块。

`engine.prediction_objectives` 保留 torch 相关的 `PredictionTargets`、`PredictionLossBundle`、`prepare_prediction_targets()`、`compute_prediction_loss()` 和具体 loss helper，并从轻量 metadata 模块复用契约表。这样可避免复制 objective 表，同时保留现有 runtime API。

备选方案是继续在 `prediction_objectives.py` 中维持所有内容，只在 config 侧延迟导入。该方案能减少文件移动，但无法从根上保证轻量导入边界，后续容易再次把 torch 带入配置加载路径。

### Decision 2: 可视化模块按职责本地化 import

`diagnostics.visualization.config` 只保留 dataclass、路径解析、模态标准化和 JSON 可序列化 helper 所需依赖。采样模块只依赖 pandas 类型标注或通用 Any、numpy 随机选择和轻量 JSON scalar helper；writers 只依赖 csv/json/path。数据集构建相关 import 只保留在 `datasets.py` 或 manifest/prediction 运行路径；matplotlib/PIL 只保留在 `render.py` 或实际 asset 渲染模块。

备选方案是创建一个可视化公共工具聚合模块承载所有依赖。该方案会减少重复 import 行，但会重新形成隐性聚合层，不符合当前按职责拆分的方向。

### Decision 3: 文档健康检查检查结构化位置而不是全文字符串

架构边界测试应读取每个 `openspec/specs/*/spec.md` 的 `## Purpose` 段落，只检查 purpose 内容是否为空、过短或等于归档占位文本。测试不应在全文中拒绝该占位字符串，否则 spec 无法描述自身的整理规则。

备选方案是改写 `project-architecture` 文本避开字符串。这只能消除当前误伤，不能防止同类自引用问题再次出现。

### Decision 4: 用 focused 测试约束边界

新增或调整 `tests/test_architecture_boundaries.py` 中的 probe：

- `import kd_sensing.config` 后 `torch`、dataset、models、diagnostics/render 栈不在 `sys.modules` 中。
- 导入 `kd_sensing.diagnostics.visualization.config`、`sampling`、`writers` 不触发 matplotlib、PIL、dataset builder 或 visualization core。
- OpenSpec purpose 检查只针对 Purpose 段落，并报告具体 spec。

这些测试不读取真实 dataset、不加载 checkpoint、不启动训练，继续使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 作为快速架构检查。

## Risks / Trade-offs

- [Risk] 拆分 objective 元数据时遗漏某个公开 helper 或 `__all__` 符号，影响训练、验证或测试导入路径。→ 保留 `engine.prediction_objectives` 的既有公开 API，并让其从轻量模块 re-export 元数据 helper；用现有 objective、Raymobtime 和架构测试回归。
- [Risk] 可视化模块移除 import 后，某些运行路径依赖隐式导入副作用。→ 分模块本地导入真实使用的依赖，并运行 viewer manifest、可视化诊断和互补 explorer 相关测试。
- [Risk] OpenSpec purpose 清理范围扩大，和当前功能实现无关。→ 本 change 只修正健康检查语义并补齐明确失败的 purpose；其它 spec 内容不重写。
- [Risk] 新增轻量模块增加一个文件边界。→ 该边界承载真实复用的 objective 契约，能减少 config 和 runtime 的反向依赖，收益大于文件数量增加。

## Migration Plan

1. 先调整 OpenSpec 文档健康检查和 purpose 文本，使架构测试的文档项不再误伤。
2. 拆分 objective 元数据，改造 config/engine 调用点，确保公开 API 兼容。
3. 清理 `diagnostics.visualization` 子模块 import，把重依赖移动到真实使用模块或函数内部。
4. 运行 focused 验证：
   - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
   - `openspec validate refine-architecture-boundaries --strict`
   - `openspec validate --all --strict`
5. 若触及 viewer 行为，再运行相关测试：
   - `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py tests/test_gradio_complementarity_explorer.py -q`

回滚策略：由于不改变公开配置和运行产物格式，若回归失败，可单独回退 objective 元数据拆分或可视化 import 清理，保留 OpenSpec 文档检查修正。

## Open Questions

- objective 轻量模块最终命名为 `kd_sensing.engine.objective_metadata` 还是顶层 `kd_sensing.objectives`。默认选择前者，以减少跨包迁移和保持 engine/objective 语义靠近。
- `openspec validate --all --strict` 当前还有多项 purpose 过短 warning；本 change 是否一次性补齐全部 purpose，还是只补齐本次触达 capability。默认优先补齐全部 purpose warning，因为它不改变需求语义且能关闭文档健康债。
