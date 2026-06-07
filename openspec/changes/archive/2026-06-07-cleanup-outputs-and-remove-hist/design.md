## Context

当前仓库已经完成 KD 退役，但项目表面仍保留另一条较大的历史研究线：HiST-Beam/Hist。它横跨 `configs/hist_beam/`、`kd-sensing-hist-beam-loso`、`src/kd_sensing/engine/hist_beam_*`、`src/kd_sensing/models/fusion/hist_beam.py`、evaluation helpers、README、docs、tests 和多个 OpenSpec specs。与此同时，`outputs/` 里积累了约 3.5G 本地运行产物，其中 `outputs/other/`、`p3_v8_*`、`v9_*`、`hist_beam_loso`、`history_anchor_*`、`image_only_legal_*`、`*_smoke`、`*_plan_check` 和 debug 目录占据主要清理面。

本 change 的用户目标是“先给方案”，并且要求整理项目、重点清理 `outputs/`、整理 `engine`/`models` 目录、删除已经不用的 Hist 模型相关代码。因此方案必须同时覆盖源码支持面、配置/文档/OpenSpec 契约和本地产物删除边界。当前工作区已有大量用户侧未提交改动；实施阶段必须只改与本 change 相关的文件，并尊重已有变更。

## Goals / Non-Goals

**Goals:**

- 退役 HiST-Beam/Hist 研究线，删除可运行 CLI、配置、模型、engine、evaluation、测试和文档入口。
- 让组件注册、配置加载、README、pyproject、docs 和 OpenSpec 不再声明 `hist_beam_fusion`、HiST variants、P3/radio prototype、image-only HiST probe 或 Hist LOSO 可用。
- 整理 `src/kd_sensing/engine` 和 `src/kd_sensing/models`，保留当前主线模块，删除 Hist 专用文件，不新增旧入口 wrapper 或兼容聚合层。
- 通过 runtime cleanup manifest 清理 `outputs/` 中明确过时的 Hist/P3/V8/V9/debug/smoke/plan-check/stale 产物，并保护当前主线、已跟踪文件、数据、源码和复现必需 checkpoint。
- 约束后续输出目录分区，使训练、analysis、cache、features、cleanup manifest 和 scene-level best checkpoints 有清晰落点。

**Non-Goals:**

- 不删除 `dataset/`、`All_models/` 已跟踪权重、OpenSpec archive、源码配置文档中的历史归档记录或当前主线输出。
- 不把裸字符串 `hist` 作为删除条件；例如 `gps_window_*hist2` 可能表示历史窗口长度，必须结合 manifest 规则、运行状态和人工可读原因判断。
- 不为旧 HiST CLI、旧配置路径或旧模型注册名提供兼容 alias。
- 不重新设计当前 GPS v2、Top8 selector、GPS+LiDAR BGAM、camera residual、CSI hardening、Raymobtime、viewer manifest 等主线 workflow。

## Decisions

### Decision 1: 直接退役 Hist 支持面，而不是保留 dormant path

实现应删除 `src/kd_sensing/cli/hist_beam_loso.py`、`configs/hist_beam/`、`src/kd_sensing/engine/hist_beam_*`、`src/kd_sensing/models/fusion/hist_beam.py`、Hist 专用 evaluation helpers、Hist training extension 入口和对应 tests/docs 引用。旧入口被引用时应通过配置加载、console script 缺失或 registry 已删除名称给出清晰失败。

替代方案是把 Hist 保留为 legacy optional workflow，但这会继续要求测试、文档、registry 和运行时分支维护，违背“现在好像用不到 hist 模型了，请删除”的目标。

### Decision 2: 输出清理必须先 manifest 后删除

`outputs/` 清理使用现有 `kd-sensing-clean-runtime-artifacts` 保护模型：先 dry-run 写 JSON manifest，记录候选路径、大小、mtime、规则 ID、原因、风险等级和保护状态；删除阶段必须显式使用 manifest 和确认参数，并重新验证路径未被 git 跟踪、未受保护、仍在允许根内。

实施时建议把候选分为：

- 高置信退役 Hist：`outputs/hist_beam_loso`、`outputs/history_anchor_*`、`outputs/image_only_legal_*`、`outputs/p3_v8_*`、`outputs/v9_*`。
- 短生命周期产物：`outputs/_debug_*`、`outputs/*_plan_check*`、资源 smoke、失败或 stale run。
- 需要复核的语义不清目录：`outputs/other/` 和命名无法映射到当前主线的根级目录。
- 默认保护：当前主线 `outputs/analysis/deepsense6g_*`、`outputs/analysis/mmw_town_gps_adapter_v2*`、`outputs/cache` 中仍被当前 workflow 引用的 cache、`outputs/features` 中当前主线 features、scene-level `best_checkpoints`、`best.pth`/`best_top1.pth` 及 sidecar metadata。

### Decision 3: 后续输出目录采用用途分区

新运行不应继续把实验直接散落在 `outputs/` 根目录或 `outputs/other/`。推荐结构：

```text
outputs/
  training/<workflow>/<scene-or-dataset>/<run_id>/
  analysis/<workflow>/<variant-or-support-ratio>/<run_id>/
  cache/<cache-family>/<fingerprint>/
  features/<feature-family>/<run_id>/
  cleanup_manifests/runtime_cleanup_<timestamp>.json
  scene<id>/best_checkpoints/
```

这不是强制迁移所有历史目录的要求；实施阶段只需要更新当前主线配置默认输出和文档说明，历史目录通过 manifest 选择性删除或保留。

### Decision 4: OpenSpec 同步退役，而不是只删代码

现有 specs 中大量要求仍写着 “HiST-Beam MUST 支持”。如果只删代码不改 specs，后续实现会与架构契约冲突。因此本 change 用 delta specs 标记 Hist 相关能力删除，并把 project、registry、workflow、cleanup 和 artifact registry 规范更新为当前支持面。

### Decision 5: 验证从结构收敛开始

验收优先覆盖“Hist 不再是支持入口”和“当前主线不被误删/误断”：

- OpenSpec：`openspec validate cleanup-outputs-and-remove-hist --strict`。
- 架构/registry：`conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_component_registry.py -q`。
- CLI：保留入口 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-export-viewer-manifest`、`kd-sensing-visualize-modalities` help 通过；`kd-sensing-hist-beam-loso` 不再作为保留入口。
- 清理：先运行 dry-run manifest，再审计候选，最后执行确认删除并保存 deletion report。
- 最终：`conda run -n kd_mm_beam pytest -q`。

## Risks / Trade-offs

- **Hist 引用分布很广，删除容易漏掉 README/docs/tests/specs 引用** -> 用 `rg -n "hist_beam|HiST|kd-sensing-hist-beam|configs/hist_beam"` 生成清单，按源码、配置、文档、测试、OpenSpec 分批收敛。
- **误删 `outputs/` 中仍有价值的当前主线结果** -> manifest 规则必须记录原因和保护状态；删除阶段只处理未保护候选，`best.pth`、sidecar、当前主线 analysis/cache/features 默认保护。
- **`hist` 字符串误伤 history-window baseline** -> 删除规则不得只靠 substring；`gps_window_*hist2` 这类目录必须按 workflow、run metadata 和用户确认原因判断。
- **移除 console script 影响旧命令习惯** -> README 和 pyproject 同步删除推荐入口；错误或缺失应指向当前支持 workflow，而不是恢复兼容 alias。
- **工作区已有大量未提交变更** -> 实施时先读取相关文件，避免 revert 非本 change 改动；只改必要文件。

## Migration Plan

1. 盘点 Hist 入口与产物：源码、配置、pyproject、README/docs、tests、OpenSpec 和 `outputs/` 候选目录。
2. 删除源码支持面：CLI、configs、engine/model/evaluation Hist 模块、registry 注册、training extension/profiling helper 和测试。
3. 更新文档和 specs：README quickstart、docs/experiment_matrix、project surface inventory、pyproject scripts/description、相关 OpenSpec specs。
4. 更新 runtime cleanup 规则或使用现有规则生成 dry-run manifest，覆盖 Hist/P3/V8/V9/debug/smoke/plan-check/stale 候选。
5. 审计 manifest 后执行删除，保存 manifest 和 deletion report 到 `outputs/cleanup_manifests/`。
6. 运行 focused 验证和全量回归。

Rollback 仅限代码层面回退本 change；删除的本地产物不设计自动恢复。若某个候选产物需要保留，应在 manifest 审计阶段标记 protected 或移出删除列表。

## Open Questions

- 是否把仓库标题 `KD for Sensing` 一并改名属于品牌层变更，不纳入本 change；本次只要求当前支持面和 README 描述不再推荐 Hist。
