## Context

当前项目已经完成了几轮重要收敛：旧 KD/HiST/BGAM/viewer/AMR mock/JEPA-MSAC 等路线被退役，默认 baseline 逐步迁到 `modular_sequence`，public CLI 入口集中到 `kd-sensing-*`，project surface inventory 与 architecture boundary tests 也能阻止旧入口回流。快速验证结果显示治理层仍健康：`openspec validate --all --strict` 通过 111 项，`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 通过 20 项。

剩余问题不是“缺少治理”，而是热点 owner 仍承担过多连接职责：

```text
config / CLI / scripts
        │
        ▼
engine.data_factory ─────────────┐
        │                         │
        ▼                         ▼
DeepSense6GDataset          MMWDataset
        │                         │
        ├── modality readers      ├── csv/column normalization
        ├── target providers      ├── geometry/radio/path semantics
        ├── scalers/cache         ├── physical labels
        └── sample assembly       └── physics supervision

training runtime
        │
        ▼
trainer._train_inner ─▶ epoch loop ─▶ evaluation pass ─▶ artifact/checkpoint/finalization

model runtime
        │
        ▼
ModularSequenceModel.forward
        ├── encoder dependency ordering
        ├── reliability metadata
        ├── token/core assembly
        ├── geometry prior / rerank
        └── diagnostics / auxiliary outputs
```

新增功能时，这些热点会被迫成为默认改动点。长期看，这会让“添加一个模态 reader、一个 target、一个 benchmark suite 或一个 representation core”变成横切修改，增加回归风险。

实施还要考虑当前工作树已有大量未提交/未跟踪实验变更。重构必须先隔离这些状态，否则无法判断失败来自实验改动还是架构移动。

## Goals / Non-Goals

**Goals:**

- 按 wave 完成全盘结构收敛，而不是只登记热点或只做局部拆分。
- 降低新增模态、target、baseline、diagnostic suite、local experiment workflow 时触碰公共热点的概率。
- 保持当前用户可见 CLI、current config 语义、dataset split、beam label / label-space、metric schema、checkpoint schema、run metadata 和默认输出分区兼容。
- 删除或合并未登记 public surface 的内部 facade、thin wrapper、低价值 `__all__`、单调用点 helper、重复小工具和纯历史 tombstone。
- 让 architecture boundary tests 继续挡住真实结构回归，同时不维护第二份完整 inventory。
- 每个 wave 独立验证，失败时能回滚到上一 wave，而不是把多个热点混成不可定位的大 diff。

**Non-Goals:**

- 不新增算法、实验 claim、模型能力或论文复现路线。
- 不改变当前训练数学语义、loss 权重默认值、数据 split、label-space mapping、checkpoint 读取口径或评估指标定义。
- 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`All_models/` 或其它本地运行产物。
- 不恢复旧入口、兼容聚合层、root-level 训练脚本、退役研究线实体 YAML 或绕过 `src/kd_sensing` 包结构的运行方式。
- 不把 Python 文件数或行数作为唯一目标；重构按 owner 职责、公开 surface、导入边界和验证覆盖判断。

## Decisions

### Decision 1: 先做状态收口，再开始源码 wave

重构前必须捕获并收口 baseline：

1. 检查 `openspec list --json`，已完成 active change 必须归档，或在本 change tasks 中记录暂缓原因。
2. 记录 `git status --short`，区分用户/前序实验变更、未跟踪配置/脚本、本地 cache 噪声和本 change 的 OpenSpec artifacts。
3. 恢复或确认 `dataset/.gitkeep`、`.gitignore`、`.codegraph/`、`.codex/skills/` 等非目标变更，避免重构误改产物边界或 agent 工具状态。
4. 运行 baseline validation：OpenSpec strict、architecture boundaries、CLI/config smoke 和目标 wave focused tests。

替代方案是直接在当前工作树上开改。该方案速度快，但会让重构 diff 与已有实验变更混杂，回滚和审查成本过高，因此拒绝。

### Decision 2: Dataset 从继承扩展转为组合式 adapter

目标结构：

```text
data/datasets/
  deepsense6g.py              # public dataset owner / thin orchestration
  sequence_dataset_core.py    # sample contract、portion、metadata、cache key
  modality_readers.py         # image/radar/gps/lidar/mmwave/csi reader 协调
  target_providers.py         # beam/soft/occlusion/position/physics targets
  dataset_family_adapters.py  # DeepSense6G / MMW family-specific setup
  sample_assembly.py          # tensor assembly / metadata assembly
```

`MMWDataset` 不再通过继承堆叠所有 MMW 特性，而是以 dataset-family adapter 方式注入 MMW 的 CSV 补列、condition layout、geometry、availability、radio/path semantic、physical label 和 physics supervision。初期可保留 `MMWDataset` class 作为 registry owner，但内部必须委托 adapter，而不是继续扩大 `__init__` / `__getitem__`。

替代方案是继续在现有类上拆更多 helper。该方案风险低，但仍把新增模态和 family-specific 行为绑定在巨型 dataset class 上，不能解决扩展默认触碰热点的问题。组合式 adapter 更适合作为长期边界。

### Decision 3: 训练与评估 runtime 使用 run context 和 phase helpers

`trainer.train(cfg)` 和 `evaluator.evaluate(cfg)` 的 public 行为保持不变。内部拆成：

```text
TrainingRunContext
  cfg / objective / run_dir / artifact_writer
  dataloaders / split_metadata / normalization_artifacts
  device / model / optimizer / scheduler / scaler
  checkpoint_manager / tensorboard / extension / state

setup phases:
  resolve_training_objective()
  prepare_training_run()
  build_training_resources()
  restore_training_state()
  run_training_loop()
  finalize_training_run()
```

`evaluation_pass` 拆成 schema-safe batch iteration、model step、objective output recording、metadata recording 和 metric aggregation。`validator`、`evaluator`、diagnostics real-forward 和 final-test evaluation 继续复用 shared evaluation pass。

替代方案是只拆 `_train_inner` 的行数。该方案可能让 helper 变成私有聚合层，因此本 change 要求每个 helper 对应真实 phase，并由 focused tests 覆盖 public schema。

### Decision 4: Modular forward 只保留 public routing，复杂阶段下沉

`ModularSequenceModel.forward` 拆为私有阶段函数或小型 internal collaborators：

- collect raw/reliability inputs
- resolve encoder dependency order
- run encoders and projectors
- assemble representation core input and availability mask
- run core/head
- apply geometry prior / safe rerank post-processing
- assemble diagnostics, runtime metadata and auxiliary outputs

新增 encoder/projector/core/head 时，默认只改组件实现和 metadata；不得继续向主 forward 堆 baseline-specific 分支。确需新增 forward-wide contract 时，必须更新 `modular-sequence-model`、`model-architecture-extension-contract` 和 focused tests。

替代方案是新增更多 whole-model exception 来避开 forward 复杂性。该方案会削弱当前组件化契约，因此只允许在 OpenSpec design 明确证明不可组合时使用。

### Decision 5: Diagnostics runner 按 suite 和 artifact family 分发

诊断入口继续保持 package CLI 和 public owner 兼容，但内部按以下边界收敛：

```text
manifest/schema       # loading, comparability, model groups
metric rows           # suite-specific condition row construction
aggregation           # robustness/shortcut/predictive/CxD/geometry summaries
artifact planning     # output registry, paths, schema
writers/plots         # CSV/JSON/NPY/PNG/SVG
real-forward          # checkpoint/config/dataloader/model execution
```

JEPA benchmark facade 只能暴露 public runner API，不重新导出 private helpers。JEPA visual analysis 如果消费 benchmark artifact，必须从 owner module 导入。

替代方案是把所有 suite 都塞回 runner。该方案会让新增 benchmark suite 继续扩大热点，和本 change 目标冲突。

### Decision 6: 配置和脚本表面转向 recipe / manifest / lifecycle

`configs/scene31` 和部分 fusion/diagnostics YAML 需要区分：

- current canonical config：长期入口，实体 YAML 保留。
- experiment reproduction/local manual config：可实体保留，但必须登记 owner、用途和输出边界。
- generated/local queue overlay：优先由 generator recipe/manifest 生成，不再无限提交实体 YAML。
- retired config：只允许 migration guard、tombstone 或历史说明，不恢复实体 YAML。

脚本分为 package CLI、数据准备脚本、研究诊断脚本、shell orchestration 和 local/manual queue helper。重复 package CLI 的 Python thin alias 必须删除；local/manual 脚本必须记录生命周期和不作为长期入口的 caveat。

替代方案是继续允许所有实验 YAML 和脚本进入 tracked surface。该方案短期方便，但会让 current 支持面越来越难读。

### Decision 7: Tombstone 只保留 guard 价值

retired-tombstone specs 分三类处理：

- 仍对应 registry/config/CLI/document wording guard：保留，并在 inventory 记录 guard 价值。
- 只剩历史叙述但仍有迁移说明价值：折叠到集中 retired summary。
- 无当前 guard、无当前引用、无迁移价值：归档。

折叠不得导致旧入口被误判为 current，也不得删除真实 guard tests。

### Decision 8: 健康护栏验证结构事实，不复制完整治理表

architecture boundary tests 保留以下硬边界：

- retired route/config/CLI/model registry 回流
- tracked runtime artifacts
- 重依赖 package barrel / facade 回流
- scripts/config/current path 引用失效
- whole-model exception 未登记
- dataset/baseline/model 依赖方向
- 普通测试重复 sys.path bootstrap
- hotspot 未登记或 public facade 变厚

测试不得维护完整源码目录清单、完整 OpenSpec prose 镜像、完整 scripts allowlist 或完整 hotspot budget 表。大型事实仍由 inventory 和 specs 维护，测试只验证可推导的结构事实和少量高风险常量。

## Risks / Trade-offs

- 大规模移动导致 import path 破坏 -> 每个 wave 先迁移内部调用方，再删除 facade；公开 CLI 和 current public owner 保持 smoke tests。
- Dataset adapter 化改变样本字段或 metadata -> 使用现有 dataset/batch focused tests，并新增 golden synthetic sample contract tests；真实 `dataset/` 不作为验证依赖。
- Training context 拆分改变 checkpoint、early stopping 或 finalization 顺序 -> 先提取无行为变化 phase，保留 `train_log.json`、`final_config.yaml`、status file 和 checkpoint schema focused tests。
- Modular forward 阶段化影响 geometry prior、rerank、reliability metadata 或 auxiliary outputs -> 以 `adapt_model_output`、metadata、diagnostics 和 representative model focused tests 锁定输出 schema。
- Diagnostics suite 拆分影响 manifest output registration -> 以 smoke manifests 覆盖 `benchmark_manifest.json`、核心 CSV/JSON/NPY 输出和 output registry。
- Tombstone 折叠过度导致旧路线回流 -> 折叠前必须证明没有 guard 价值，保留 retired route summary 和 architecture boundary guard。
- 配置 recipe 化降低人工可见性 -> 保留 manifest、生成命令、dry-run 输出和 sanity tests；current canonical config 不生成化。
- 当前脏工作树影响实施 -> Wave 0 必须先收口或明确 deferral；重构提交不得混入无关实验输出或本地运行产物。

## Migration Plan

1. **Wave 0: 状态收口与 baseline**  
   归档或记录暂缓 complete changes，清理/隔离未跟踪实验表面，恢复产物边界占位文件，运行 baseline validation，更新 inventory 中本 campaign baseline。

2. **Wave 1: Dataset contract adapter 化**  
   提取 sequence dataset core、resource reader coordination、target provider、sample assembly、family adapter。保持 `DATASETS.build({"type": "deepsense6g"})` 和 `{"type": "mmw"}` 行为兼容。

3. **Wave 2: Training/evaluation runtime 分层**  
   引入 run context 和 setup/finalization phases，拆分 evaluation pass schema helpers，保持 train/evaluate CLI、checkpoint 和 metric outputs 兼容。

4. **Wave 3: Modular forward 阶段化**  
   拆 `ModularSequenceModel.forward` 内部阶段，保留 public signature 和 output keys，补 metadata/diagnostics focused tests。

5. **Wave 4: Diagnostics runner 收敛**  
   拆 JEPA benchmark runner、JEPA visual analysis、MMW GPS v2、run index 和 cleanup/organize manifest 的 suite/artifact responsibilities；保持 CLI 和 manifest schema。

6. **Wave 5: Config/script/import surface 收敛**  
   配置 recipe/manifest 化，脚本 lifecycle 分类，删除低价值 facade/thin wrapper/`__all__`，迁移内部 imports 到 owner modules。

7. **Wave 6: OpenSpec/docs/guardrails 收口**  
   折叠无 guard 价值 tombstone，更新 project surface inventory、agent navigation、focused validation commands 和 architecture boundary tests。

8. **Final validation**  
   运行 `openspec validate streamline-project-architecture-waves --strict`、`openspec validate --all --strict`、所有 wave focused tests、CLI help smoke 和 `conda run -n kd_mm_beam pytest -q`。若全量测试因环境或本地数据缺失无法完成，最终说明必须列出已完成替代验证和剩余风险。

Rollback 策略以 wave 为单位：任一 wave 行为验证失败且无法快速修复时，回退该 wave 的源码移动，保留已通过 wave；不得用恢复旧 facade 或旧入口作为长期修复。

## Open Questions

- 当前已完成但未归档的 `align-amber-amr-paper-architectures` 和 `add-scene31-next-round-experiments` 是先归档，还是在本 change 的 Wave 0 中记录 deferral 并继续保留 active？
- `configs/scene31` 中哪些未跟踪 next-round/night-grid YAML 应进入长期 recipe，哪些只是本地队列产物不应提交？
- `MMWDataset` 是否在本 change 中完全取消继承，还是先保留 subclass public owner、内部委托 adapter，待后续 change 再删除继承？
- 哪些 retired tombstone 已经没有 guard 价值，需要在实施前逐项审计 registry/config/CLI/docs/tests 引用后决定。
