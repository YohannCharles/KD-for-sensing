## Why

`outputs/` 现在同时承载 cache、best checkpoint、训练 run、评估 run、诊断分析和历史临时产物，并且命名横跨 `outputs/<run_name>`、`outputs/31`、`outputs/scene31`、`outputs/eval_*`、`outputs/evaluations` 等多套时代规则。继续让新实验写入这些混合根目录，会让 checkpoint 解析、run index、复现实验对比和人工清理都越来越容易误判。

## What Changes

- 定义当前支持 workflow 的 canonical runtime output layout：cache、cleanup manifest、analysis、visual analysis、evaluations、单场景训练 run、multi-scene/scenegroup 训练 run、scene/scenegroup best checkpoint registry 和 legacy archive 分区各自有固定语义。
- 修改训练/评估默认输出规则：单场景 DeepSense6G run 继续写入 `outputs/<scene_slug>/<run_name>/`；多场景协议 MUST 写入稳定的 `outputs/scenegroup_<scene-list>/<run_name>/` 或配置显式指定的等价根目录；默认评估集合 SHOULD 写入 `outputs/evaluations/<study_id>/`。
- 扩展 checkpoint registry：scene-level registry 继续按 `outputs/scene*/best_checkpoints/` 隔离；multi-scene run 使用 scenegroup registry；根级 `outputs/best_checkpoints/` 只作为 legacy 输入由迁移/审计 manifest 处理，不再作为当前默认写入目标。
- 新增 runtime output 整理 manifest 能力：先只读生成 move/archive plan，记录源路径、目标路径、类型、大小、mtime、冲突、引用风险和是否需要人工确认；执行阶段必须显式确认，不得静默移动或删除本地产物。
- 更新 run index 和 cleanup 扫描边界：默认避免递归扫描大体量 `outputs/cache/` 拖慢 run 索引；cleanup 继续保护 current mainline 分区，并能识别 `outputs/archive/`、legacy root run、legacy `eval_*` 和 numeric scene 目录。
- 同步 README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md` 和相关配置引用，避免当前文档继续推荐混乱的 legacy 输出路径。
- 不自动删除 `dataset/`、`outputs/cache/`、历史 checkpoint、TensorBoard 或真实训练输出；迁移执行必须经过 manifest 和显式确认。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-surface-cleanup`: 输出目录用途分区从原则性约束细化为 canonical taxonomy、legacy archive 和文档同步要求。
- `experiment-workflow`: 训练和评估输出规则增加 multi-scene/scenegroup slug、默认 evaluations 分区和 legacy 根目录迁移约束。
- `experiment-artifact-registry`: best checkpoint registry 增加 multi-scene/scenegroup 隔离，并明确根级 `outputs/best_checkpoints` 的 legacy 处理方式。
- `runtime-artifact-cleanup`: 清理 manifest 扩展为整理/迁移 manifest，支持只读 move/archive plan、冲突检测和显式执行阶段。
- `experiment-run-index`: run index 必须理解 canonical layout，并默认避免把 cache/archive 当作普通 run 根递归扫描。

## Impact

- 影响代码：`kd_sensing.engine.trainer` 输出根解析、`kd_sensing.data.scenes` 或等价 scene/scenegroup 元数据 helper、`kd_sensing.utils.artifact_registry` registry 目录解析、`kd_sensing.diagnostics.run_index` 扫描过滤、`kd_sensing.diagnostics.runtime_artifact_cleanup` manifest 规则，以及可能新增的 runtime output migration helper/CLI。
- 影响配置和文档：当前 JEPA、vision-position、BeamBench/Arnold22、visual analysis、evaluation 示例中的输出路径和 checkpoint 引用需要同步；`docs/experiment_matrix.md` 和 inventory 需要记录新 layout。
- 影响验证：需要运行 OpenSpec strict 校验、架构边界测试、run index/cleanup focused tests、训练 dry-run 或输出目录单元测试，以及 CLI help smoke。所有项目相关 Python 命令必须使用 `conda run -n kd_mm_beam ...`。
- 不影响：不移动真实 `dataset/` 数据，不提交 `outputs/` 产物，不删除历史本地 checkpoint，不改动模型数值逻辑。
