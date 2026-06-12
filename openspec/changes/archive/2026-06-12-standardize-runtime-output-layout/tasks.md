## 1. Layout 与 Scope 基础

- [x] 1.1 新增或扩展 runtime output layout helper，集中定义 `cache`、`cleanup_manifests`、`analysis`、`visual_analysis`、`evaluations`、`scene<id>`、`scenegroup_*` 和 `archive` 分区。
- [x] 1.2 实现 DeepSense6G scene/scenegroup scope 推导，覆盖单 scene、连续多 scene、非连续多 scene、`train_scenes`、`validation_scenes`、`test_scenes` 和 `eval_scenes`。
- [x] 1.3 将 scene/scenegroup scope 写入 final config runtime metadata，并保持现有单场景 metadata 兼容。
- [x] 1.4 添加 layout helper 单元测试，使用 `conda run -n kd_mm_beam pytest <focused-test> -q` 验证 slug、路径和显式输出覆盖行为。

## 2. 训练、评估与 Registry 输出

- [x] 2.1 修改训练 run 目录解析：单场景默认写入 `outputs/scene<id>/<run_name>/`，多场景默认写入 `outputs/scenegroup_*/<run_name>/`。
- [x] 2.2 修改 resume 路径解析，确保 `training.resume=true` 只从当前 scene/scenegroup scope 下的 `checkpoints/last.pth` 恢复。
- [x] 2.3 修改默认评估输出规则，使成组评估写入 `outputs/evaluations/<study_id>/` 或当前 scope 下的 evaluation run，显式 `--output-dir` 保持完整路径。
- [x] 2.4 修改 checkpoint registry 默认目录解析，支持 scene 和 scenegroup registry，并停止把根级 `outputs/best_checkpoints/` 作为当前默认写入目标。
- [x] 2.5 添加训练/评估/registry focused tests，使用 `conda run -n kd_mm_beam pytest <focused-tests> -q` 验证单场景、多场景、显式输出目录和 registry 隔离。

## 3. Run Index 与整理 Manifest

- [x] 3.1 修改 run index 默认扫描过滤，跳过 `outputs/cache/`、`outputs/archive/`、`outputs/cleanup_manifests/` 等非 run 分区，同时保留显式扫描这些根的能力。
- [x] 3.2 扩展 run summary，记录 scene scope、scenegroup scope、canonical partition 和 legacy/archive 标记。
- [x] 3.3 新增 runtime output organize dry-run manifest，输出 move/archive/protect/review plan，不移动、不删除、不重写任何本地产物。
- [x] 3.4 实现 legacy 分类规则，覆盖根级训练 run、`outputs/31/`、根级 `outputs/best_checkpoints/`、`outputs/eval_*`、cache summary 和无法判定 scope 的人工复核项。
- [x] 3.5 如提供整理执行阶段，实现显式确认、source 状态复核、target 冲突检查、git tracked 保护和 execution report。
- [x] 3.6 添加 run index 与 organize manifest focused tests，使用 `conda run -n kd_mm_beam pytest <focused-tests> -q` 覆盖跳过 cache、legacy 分类、冲突跳过和未确认拒绝执行。

## 4. 配置与文档同步

- [x] 4.1 更新当前 JEPA、vision-position、BeamBench/Arnold22、visual analysis 和 evaluation 示例配置中的输出路径与 checkpoint 引用。
- [x] 4.2 更新 README 的 outputs 目录说明、run index 示例、cleanup/organize manifest 示例和当前推荐实验路径。
- [x] 4.3 更新 `docs/experiment_matrix.md`，将训练、评估、analysis、cache、registry 和 legacy/archive 的路径规范写入实验矩阵说明。
- [x] 4.4 更新 `docs/project_surface_inventory.md` 和架构 guardrail，拒绝新增默认 `outputs/other`、根级 `outputs/<run_name>`、数字场景根和根级 best checkpoint 写入。
- [x] 4.5 确认 historical/archive 文档不被误标为当前推荐入口。

## 5. 验证

- [x] 5.1 运行 `openspec validate standardize-runtime-output-layout --strict`。
- [x] 5.2 运行 `openspec status --change standardize-runtime-output-layout` 并确认 apply-ready。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_runtime_artifact_cleanup.py -q` 或新增 focused tests 的等价集合。
- [x] 5.5 运行 `conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs --format json --no-resources --output outputs/analysis/run_index_smoke.json`，确认默认扫描不会被 cache 拖慢。
- [x] 5.6 如新增 organize CLI，运行 `conda run -n kd_mm_beam <organize-entrypoint> --help` 和 dry-run smoke，并确认不会移动或删除 `outputs/` 产物。
