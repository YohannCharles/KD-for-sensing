## Context

当前 `outputs/` 的真实状态暴露出三类问题：

- 语义混合：cache、best checkpoint、训练 run、评估 run、分析图表、cleanup manifest 和历史临时产物平铺在同一层。
- 时代规则并存：既有 `outputs/<run_name>/`、`outputs/31/`，也有较新的 `outputs/scene31/`、`outputs/evaluations/`、`outputs/training/`。
- 工具边界不一致：训练默认按 scene 分组，部分多场景配置关闭 `group_by_scene` 后写回根目录；checkpoint registry 已有 scene 目录逻辑，但根级 `outputs/best_checkpoints/` 仍存在；run index 扫全量 `outputs/` 时会被大体量 `outputs/cache/` 拖慢。

本 change 面向本地 ignored runtime artifacts，不改变模型结构、训练数值逻辑或数据集真实目录。

## Goals / Non-Goals

**Goals:**

- 定义一套可读、可测试、可迁移的 runtime output taxonomy。
- 让单场景、多场景、评估、分析、cache、best checkpoint registry、cleanup/organize manifest 和 legacy archive 各有固定位置。
- 让训练、评估、registry、run index、cleanup/migration manifest 和文档共享同一套 layout 口径。
- 用 manifest 驱动历史整理：先审计、再显式确认执行，避免误动 checkpoint、cache 或仍有引用的 run。
- 修复多场景配置只能靠 `group_by_scene: false` 写根目录的问题，引入稳定的 scenegroup slug。

**Non-Goals:**

- 不自动删除 `dataset/`、`outputs/cache/`、`logs/`、checkpoint、TensorBoard event 或历史训练输出。
- 不把本地 `outputs/` 纳入源码提交。
- 不重训模型，不重算指标，不修改模型权重内容。
- 不在本轮重构所有历史脚本的实验矩阵语义；只修当前支持面和推荐入口。
- 不把 legacy archive 作为新的运行入口或默认 checkpoint 解析来源。

## Decisions

### 1. 保留 scene-level run 根，同时新增 scenegroup 根

单场景 DeepSense6G 训练已经由 `create_run_dir()` 写入 `outputs/<scene_slug>/<run_name>/`，且 OpenSpec 当前契约也要求默认 Scenario 31 写入 `outputs/scene31/<run_name>/`。直接改成 `outputs/training/...` 会造成大面积引用漂移。

因此本 change 选择：

```text
outputs/
  scene31/<run_name>/
  scene32/<run_name>/
  scenegroup_s32_s34/<run_name>/
  scenegroup_s31_s34/<run_name>/
```

`outputs/training/` 可以保留为历史或特定 workflow 显式根，但不作为通用训练默认根。

替代方案：所有训练 run 统一迁入 `outputs/training/<scope>/<run_name>/`。该方案层级更规整，但会与现有 scene registry、resume 路径和 README 口径冲突更大，本轮不采用。

### 2. 多场景 scope 从数据 split 推导，而不是默认伪装成单 scene

当配置包含 `train_scenes`、`validation_scenes`、`test_scenes` 或 `eval_scenes`，且这些集合不等同于单个 `data.dataset.scene` 时，系统需要生成稳定的 scenegroup slug。建议规则：

- 只覆盖 scene 32/33/34：`scenegroup_s32_s34`
- 覆盖 scene 31/32/33/34：`scenegroup_s31_s34`
- 其它不连续组合：按升序拼接，如 `scenegroup_s31_s33`

slug 写入 final config runtime metadata，供 registry、run index 和文档引用。

### 3. best checkpoint registry 与 run scope 对齐

当前 scene-level registry 保留：

```text
outputs/scene31/best_checkpoints/
outputs/scene32/best_checkpoints/
```

多场景 registry 使用：

```text
outputs/scenegroup_s32_s34/best_checkpoints/
outputs/scenegroup_s31_s34/best_checkpoints/
```

根级 `outputs/best_checkpoints/` 不再作为当前默认写入目标。历史文件通过整理 manifest 审计后进入 legacy archive，或者在确认 metadata 匹配时迁入 scene/scenegroup registry。

### 4. 评估集合统一到 `outputs/evaluations`

当前存在大量 `outputs/eval_*` 根目录。新默认评估集合使用：

```text
outputs/evaluations/<study_id>/<model_or_variant>/<split_or_scene>/
```

显式 `--output-dir` 仍保持完整路径，不额外追加 scene/scenegroup。历史 `outputs/eval_*` 通过 manifest 迁入 `outputs/archive/legacy_eval_runs/` 或 `outputs/evaluations/legacy/<study_id>/`，由 manifest 记录选择。

### 5. 整理 manifest 与删除 manifest 分层

已有 cleanup manifest 偏向删除候选。outputs 规范化需要 move/archive plan，不能复用“删除候选”等价处理。

新增或扩展 manifest schema 时，记录：

- `action`: `move`、`archive`、`protect`、`review`
- `source_path` 和 `target_path`
- `artifact_type`: training_run、evaluation_run、registry_checkpoint、cache、analysis、legacy_numeric_scene、legacy_root_run 等
- size、mtime、run state、checkpoint retention 摘要
- conflict 状态和 requires_manual_review 标记
- 引用风险，例如 config/checkpoint sidecar 中仍指向旧路径

执行阶段要求显式确认，并在执行前重新验证路径未变化、未被 git 跟踪、仍在允许根下且 target 不冲突。

### 6. run index 默认跳过非 run 分区

run index 仍能在用户显式指定时扫描任意根，但默认扫描 `outputs/` 时应跳过大体量或非 run 分区：

- `outputs/cache/`
- `outputs/archive/`
- `outputs/cleanup_manifests/`
- 可配置的其它 ignored partitions

这样 `kd-sensing-runs --outputs outputs` 不会因为 cache 递归变慢，也不会把 archive 当作当前活跃 run 列表混在一起。

## Risks / Trade-offs

- 路径引用漂移 → migration manifest 必须记录旧路径引用，文档和配置引用同步更新；保留 legacy archive 不作为默认解析源。
- 多场景 slug 推导出错 → 针对 `train_scenes/test_scenes`、连续/非连续 scene 组合、`group_by_scene: false` 增加 focused tests。
- 历史 root checkpoint 迁错 scope → 默认将根级 `outputs/best_checkpoints` 标为 review，只有 sidecar metadata 清晰且目标不冲突时才可自动建议迁移。
- run index 跳过 cache 后漏掉误写入 cache 的 run → 默认跳过只适用于标准扫描；用户显式 `--outputs outputs/cache` 时仍可扫描。
- 执行迁移破坏外部笔记中的路径 → manifest 执行前生成 report，并保留 archive 路径；README 明确历史路径可能需要从 manifest 追踪。

## Migration Plan

1. 增加 runtime output layout helper：解析 scene/scenegroup scope、canonical partitions 和 legacy archive targets。
2. 修改训练、评估、checkpoint registry 默认路径，保持显式 `output.dir` 和 `--output-dir` 优先级。
3. 扩展 run index 默认扫描过滤，避免 cache/archive/manifest 分区拖慢或污染结果。
4. 新增整理 manifest dry-run：只读扫描 `outputs/`，生成 move/archive/protect/review plan。
5. 更新 README、`docs/experiment_matrix.md`、inventory 和相关配置路径引用。
6. 运行 strict 校验和 focused tests。需要真实迁移时，先让用户审阅 manifest，再显式执行。

Rollback 策略：源码变更可通过 git 回滚；本地产物迁移一旦执行，必须依赖 manifest/report 中的 source/target 映射反向移动，因此执行阶段必须写出完整 report。

## Open Questions

- 是否把 `outputs/visual_analysis/` 作为长期独立分区保留，还是迁入 `outputs/analysis/jepa_visual_analysis/`？本 proposal 暂时允许保留现状，但文档需统一口径。
- 对于 `outputs/training/` 中已有历史 workflow，是保留为显式 workflow 根，还是归入 archive？实现前可由整理 manifest 先分类，不直接删除。
