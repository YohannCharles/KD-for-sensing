## Context

当前仓库已经有较清晰的第一层边界：dataset/model/distillation/engine/config/diagnostics 分包，registry 延迟导入，builder 和 transform 也已经从旧 facade 中迁出。问题转移到了第二层增长：`engine.trainer.train()` 仍然同时承担运行目录创建、模型/teacher 初始化、训练 batch step、extension 调度、objective loss、running metrics、validation 聚合、early stopping、checkpoint、TensorBoard、CSI/LiDAR diagnostics 和最终 artifact 写出；`config/io.py` 也同时承担 YAML 加载、virtual config、命令行覆盖、objective 默认值、模态推导、dataset 专属规则、迁移拒绝和 schema 校验。

这次重构的目标不是改变算法，而是让训练编排和配置编排拥有可持续的模块边界。所有 Python 验证命令继续使用 `conda run -n kd_mm_beam ...`，并且本变更不得提交 `outputs/`、`logs/`、cache、checkpoint 或真实数据。

## Goals / Non-Goals

**Goals:**

- 保留 `kd_sensing.engine.trainer.train`、`kd-sensing-train`、`scripts/train.py` 的用户入口和输出语义。
- 将训练运行时状态、batch step、epoch metric/history、checkpoint/sidecar、TensorBoard 和最终 artifact 写出拆到窄模块。
- 将 `config/io.py` 收敛为入口协调器，把 normalization、validation、dataset-specific rules 和迁移拒绝逻辑拆到独立 helper。
- 用 characterization tests 锁定重构前后的训练输出 key、checkpoint metadata、TensorBoard tag、CLI help 和 config 解析结果。
- 修复可见的 CLI/spec/docs 漂移：`kd-sensing-visualize-modalities` 若继续被 spec 或文档承诺，必须恢复为薄兼容入口并纳入 help 测试。

**Non-Goals:**

- 不改变模型结构、forward 输出契约、state_dict key、checkpoint registry 格式或默认 checkpoint 选择策略。
- 不改变 G2D、CRAF、MARF、Raymobtime selection、多任务 objective 或 CSI hardening 的 loss 数值定义。
- 不引入 Hydra、Lightning、Accelerate 等外部训练框架。
- 不删除现有实体 YAML、canonical config 路径或当前实验输出目录结构。
- 不在本变更内清理历史 `outputs/`、`logs/`、`dataset/` 或已忽略的本地产物。

## Decisions

### 1. `train()` 保留公开入口，内部改成运行时编排器

`train(cfg)` 继续是唯一公开训练入口，但内部拆出以下职责对象或 helper：

```text
train(cfg)
  ├─ TrainBootstrap        # seed/runtime threads/run_dir/dataloaders/device/model/optimizer
  ├─ TrainingState         # epoch、best metric、resume、history、checkpoint refs
  ├─ BatchStepRunner       # prepare batch、forward、base KD loss、objective loss、backward、optimizer step
  ├─ EpochMetricsRecorder  # running loss、history、epoch_log、objective fields
  ├─ CheckpointManager     # best/last/top1 checkpoint、sidecar、registry archive
  ├─ TensorBoardLogger     # startup scalar、objective scalar、method scalar
  └─ ArtifactWriter        # train_log、training_outputs、final_config、curves、debug artifacts
```

替代方案是拆成 `BaseTrainer`、`G2DTrainer`、`RaymobtimeTrainer` 等多个 class。这个方案会复制 checkpoint、AMP、scheduler、logging 和 validation 逻辑，短期更容易行为漂移。因此本次选择“单生命周期 + 窄职责 helper”。

### 2. Batch step 是最小可迁移单元

优先把 batch 内部流程迁出，而不是先重排整个 epoch：

```text
raw_batch
  -> prepare_task_batch / labels / auxiliary targets
  -> extension.before_forward
  -> run_model_step
  -> base KD/no-KD loss
  -> extension.after_forward
  -> compute_prediction_loss
  -> backward / grad clip / optimizer step
  -> scalar diagnostics
```

这样能让 `trainer.py` 先减少最复杂的一段，同时保留现有 extension API。CRAF/MARF/G2D 特有逻辑继续位于对应 extension；BatchStepRunner 只负责通用训练 step 编排。

### 3. 训练日志和输出兼容由专门 recorder 负责

当前 `trainer.py` 手写大量 `running_*`、`history[...]` 和 `epoch_log` 字段。新方案引入 `EpochMetricsRecorder` 或等价 helper，消费 objective metadata 和 loss bundle，集中生成：

- `history` 数组，包括历史兼容字段和 active objective 字段。
- `epoch_log`，包括 optimizer param groups、teacher prior、health diagnostics、validation subset metrics 和 objective metadata。
- `training_outputs.npz` payload。
- TensorBoard objective scalar 映射。

验收标准不是文件变短，而是新增 objective 或修改 validation metric 时，不需要在 trainer 主循环里追加新的 alias/history/TensorBoard 表。

### 4. Checkpoint 和 artifact 写出集中管理

`best.pth`、`best_top1.pth`、`last.pth`、sidecar、checkpoint registry archive、`teacher_metrics.json`、`train_log.json`、`final_config.yaml` 和 debug artifacts 将由 `CheckpointManager` 与 `ArtifactWriter` 协调。它们必须保留现有 key、路径和兼容 fallback：

- `best.pth` 继续代表 early stopping primary metric。
- `best_top1.pth` 继续为历史 Top-1 辅助选择。
- `last.pth` 继续保存 resume 所需状态。
- 历史 checkpoint 缺少通用 early stopping metadata 时继续走兼容恢复路径。

替代方案是把 checkpoint 写出留在 trainer，只迁出 batch step。这样主循环仍会在每个新 artifact 需求下膨胀，不能解决当前问题。

### 5. 配置加载改为 source -> overlay -> normalize -> validate pipeline

`config/io.py` 保留 `load_config`、`dump_config`、`parse_overrides` 等入口，但不再直接承载所有规则。建议拆分为：

```text
config/io.py
  ├─ load_config_source(path)          # 实体 YAML 或 virtual config
  ├─ apply_cli_overrides(cfg, overrides)
  ├─ normalize_config(cfg, context)    # objective、modalities、image/lidar、scene、snapshot
  └─ validate_loaded_config(cfg)       # 通用 schema + dataset-specific rules

config/normalization.py
config/validation.py
config/dataset_rules/raymobtime.py
config/dataset_rules/deepsense.py
config/migration_guards.py
```

其中 canonical virtual config 继续由 `config/canonical.py` 和 `canonical_recipes/` 负责；Raymobtime future/history 禁用规则进入 dataset rule helper；removed image motion、legacy option 拒绝进入 migration guard。`config/io.py` 可以协调这些 helper，但不手写具体业务规则。

### 6. 兼容可视化入口恢复为薄 alias

当前 spec 和工具文档仍提到 `kd-sensing-visualize-modalities`，但环境中的脚本会导入不存在的 `kd_sensing.cli.visualize_modalities`。本变更选择恢复薄兼容入口：

- 新增或恢复 `kd_sensing.cli.visualize_modalities:main`。
- 该入口不复制 parser 或实现，只委托 `kd_sensing.cli.export_viewer_manifest.main`，并在 help/description 中说明推荐入口是 `kd-sensing-export-viewer-manifest`。
- `pyproject.toml` 的 `[project.scripts]` 与 spec 保持一致。

替代方案是从 spec、README 和工具文档中删除该入口承诺。考虑到当前环境已经存在 stale console script，恢复薄 alias 对用户更稳，也符合已有 project-architecture 中的兼容入口场景。

### 7. 架构测试关注职责回退，不设僵硬行数阈值

新增测试应验证职责边界，而不是简单限制文件行数。可行检查包括：

- `trainer.py` 不新增 G2D/CRAF/MARF/Raymobtime 方法特有大段 helper。
- `config/io.py` 不直接包含 Raymobtime future/history 禁用 token 扫描、removed image path 拒绝等业务细节。
- CLI help 覆盖 `kd-sensing-export-viewer-manifest` 和兼容 `kd-sensing-visualize-modalities`。
- 重构后 synthetic/fixture 短训练的输出 key 与 characterization fixture 匹配。

行数可以作为诊断信息，但不作为第一版硬失败条件，避免测试鼓励机械拆文件。

## Risks / Trade-offs

- [Risk] 训练输出字段遗漏或顺序变化导致现有分析脚本失效。  
  Mitigation: 先补 characterization tests，锁定 `train_log.json`、`training_outputs.npz`、checkpoint sidecar 和 TensorBoard tag。

- [Risk] BatchStepRunner 抽象过宽，变成新的小型 trainer。  
  Mitigation: 它只处理单 batch 通用流程；epoch、checkpoint、artifact 和 method-specific extension 逻辑留在各自模块。

- [Risk] 配置 pipeline 拆分后默认值应用顺序变化。  
  Mitigation: 为实体配置、virtual canonical 配置、Raymobtime 配置和 snapshot 配置分别做 load_config 等价测试。

- [Risk] 恢复兼容可视化入口延长旧入口生命周期。  
  Mitigation: 该入口只作为薄 alias，文档继续推荐 `kd-sensing-export-viewer-manifest`；架构测试禁止复制 parser/main 实现。

- [Risk] 与未归档的 `add-raymobtime-s008-selection` active change 产生上下文噪声。  
  Mitigation: 实施前优先归档已完成 Raymobtime change，或在实现任务中明确只改架构治理范围。

## Migration Plan

1. 补齐 characterization tests：训练日志字段、checkpoint metadata、TensorBoard tag、config load 结果、CLI help。
2. 恢复 `kd-sensing-visualize-modalities` 薄 alias，修正 `pyproject.toml`、文档和架构测试。
3. 提取 `TrainingState`、`EpochMetricsRecorder` 和 `CheckpointManager`，保持 `train()` 行为不变。
4. 提取 `BatchStepRunner`，让 trainer 主循环只调用单 batch step 并消费返回的 loss/diagnostics。
5. 提取 `ArtifactWriter` 和 TensorBoard logger，保持输出路径和 key 兼容。
6. 拆分 config pipeline helper，把 `config/io.py` 中的 normalization、validation、dataset rule 和 migration guard 分阶段迁出。
7. 运行 focused tests：architecture boundaries、training IO、prediction objectives、Raymobtime selection、viewer manifest/CLI help、canonical config resolution。
8. 运行 `openspec validate refactor-training-config-orchestration --strict`、`openspec status --change refactor-training-config-orchestration` 和最终 `conda run -n kd_mm_beam pytest -q`。

Rollback 策略：每一步都保持公开入口和输出格式兼容；若某个 helper 迁移引入行为漂移，可保留新 helper 但让 `train()` 或 `load_config()` 暂时回退到旧路径，直到 characterization test 对齐。

## Open Questions

- `TrainingState`、`EpochMetricsRecorder`、`CheckpointManager` 是否放在单个 `engine/training_state.py`，还是拆成 `engine/history.py`、`engine/checkpointing.py` 和 `engine/artifacts.py` 三个文件？建议实现时按测试和依赖方向决定，避免过早拆太碎。
- `config/dataset_rules/` 是否做成包，还是先用 `config/validation.py` 中的窄函数承载？建议 Raymobtime/DeepSense 规则超过一个文件职责后再建包。
- 是否需要在本变更内清理 archived specs 的 `Purpose: TBD`？这属于文档卫生，建议另起小 change，避免扩大本次重构范围。
