## Why

当前项目的主线、baseline、实验协议和结果口径已经散布在 README、`docs/experiment_matrix.md`、配置族 README、OpenSpec specs、`BASELINE_REPORT.md` 和运行流水账中；维护者能读懂，但新读者和 AI agent 容易把历史 ablation、smoke/lowmem 配置或退役路线误解为当前推荐入口。

现在需要把“当前可引用事实”和“历史运行记录”分层，建立主线模型/实验目录、实验参数口径表和结果账本，并清理 current spec 内部的旧 KD/teacher/student 漂移，避免文档继续承载互相矛盾的实验叙事。

## What Changes

- 新增主线实验文档治理能力，要求维护当前主线模型目录、实验协议/参数表、结果/claim 账本和跨文档链接规则。
- 将 baseline 报告拆分为 current summary 与 historical log 两层，明确官方复现、本地 substitute、strict-validation、upper-bound、smoke/mock 和历史 ablation 的状态。
- 为主要配置族补齐或统一实验协议说明，包括 JEPA Image+GPS、BeamBench/Arnold22 Camera AE+GPS Direct、BEV-Fusion 2604、MMW/DeepSense BGAM、MMW GPS v2、CSI hardening 和 difficulty/benchmark profile。
- 清理 `experiment-workflow` 中仍把 teacher/student、KD mode 或旧 no-KD 路径描述为当前工作流的陈旧需求，收敛到 `model.primary`、supervised/adaptation、JEPA、BGAM、CSI hardening 和当前 baseline/control。
- 增强健康护栏，使 current docs/specs 不能重新把退役、supporting、historical、mock 或 upper-bound 口径写成当前推荐入口或正式结果。
- 不新增模型、不改变训练/评估数值语义、不提交真实数据、checkpoint、cache、metrics 或运行产物。

## Capabilities

### New Capabilities
- `mainline-experiment-documentation`: 维护当前主线模型目录、实验协议/参数表、结果账本、baseline 报告分层和跨文档索引规则。

### Modified Capabilities
- `experiment-workflow`: 将当前实验 workflow 规格从旧 KD/teacher/student 叙事收敛到当前 `model.primary` 与 supervised/adaptation/JEPA/BGAM/CSI/diagnostic workflow，并要求实验文档声明正式、smoke、lowmem、upper-bound、历史 ablation 等运行状态。
- `beambench-baseline-reproduction`: 强化 BeamBench/Arnold22 baseline 报告边界，要求 current summary 与历史流水账分离，并将 `beam_target_source=current` 作为 Table III 本地 substitute 的当前口径。
- `project-health-guardrails`: 增加文档和 OpenSpec 漂移检查，防止 current docs/specs 把退役路线、旧 KD wording、mock/smoke、upper-bound 或历史 ablation 描述为当前正式结果。
- `spec-lifecycle-boundaries`: 明确 current spec 内部不得同时保留互相冲突的 active 与 retired/supporting 语义；发生冲突时必须通过 change 清理为单一 lifecycle 叙事。

## Impact

- 主要影响 `openspec/specs/experiment-workflow/spec.md`、`openspec/specs/beambench-baseline-reproduction/spec.md`、`openspec/specs/project-health-guardrails/spec.md`、`openspec/specs/spec-lifecycle-boundaries/spec.md` 和新增 `openspec/specs/mainline-experiment-documentation/spec.md`。
- 预计实现会新增或重排文档，如 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`，并更新 README、`docs/experiment_matrix.md`、`docs/project_surface_inventory.md`、`README_REPRODUCE.md`、`BASELINE_REPORT.md` 和 `results/reproduce_baseline.md` 的引用关系。
- 可能新增轻量静态检查或扩展架构边界测试，用于验证文档索引、OpenSpec lifecycle 分类、旧口径 wording 和本地产物边界；所有 Python 检查必须使用 `conda run -n kd_mm_beam ...`。
- 不影响公开 CLI、模型 registry、训练配置解析、dataset contract、checkpoint schema 或实际训练/评估输出。
