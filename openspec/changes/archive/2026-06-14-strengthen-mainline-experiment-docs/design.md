## Context

项目已经完成多轮表面积收敛：当前 README、`docs/project_surface_inventory.md` 和 OpenSpec lifecycle inventory 都把主线指向 Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、viewer manifest、JEPA visual analysis 和通用训练评估能力。

问题不在于缺少配置或入口，而在于当前事实散落在多个层级：

- README 负责 quickstart 和高层入口，但已经承载较多 MMW/BGAM/JEPA/diagnostic 说明。
- `docs/experiment_matrix.md` 负责实验矩阵，但同时包含协议、命令、结果、caveat 和退役说明。
- `configs/**/README.md` 与 YAML 注释保存了重要参数真相，但不适合做跨模型横向比较。
- `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md` 既有 current 结论，也有历史 ablation 和旧命令。
- `openspec/specs/experiment-workflow/spec.md` 内部仍有旧 teacher/student/KD 场景，与后续去 KD 化要求冲突。

本 change 采用“当前事实层 + 历史日志层 + 规格护栏层”的方式收束，不改变训练、评估和模型实现。

## Goals / Non-Goals

**Goals:**

- 建立主线模型目录，能一眼看出每条当前主线或 baseline 的用途、配置、模型类型、数据场景、split、target、metric、状态和对照关系。
- 建立实验协议/参数表，明确 formal、lowmem、smoke、debug、upper-bound、historical ablation、mock 等运行状态。
- 建立结果/claim 账本，记录可引用结果的 config、commit/日期、数据口径、metric、checkpoint provenance、是否官方复现、是否本地 substitute，以及不能声称的范围。
- 将 BeamBench/Arnold22 报告拆成 current summary 和 historical log，降低误抄旧 `future` target 命令或 upper-bound 数值的风险。
- 清理 current OpenSpec 中仍残留的旧 KD/teacher/student active wording。
- 用轻量检查或架构边界测试约束文档索引、lifecycle 分类和退役/历史 wording。

**Non-Goals:**

- 不新增模型、loss、dataset、CLI 或训练 workflow。
- 不重新跑真实训练，不提交 checkpoint、metrics、cache、TensorBoard 或 `outputs/` 产物。
- 不把 historical local run 数值提升为官方复现结论。
- 不删除历史报告内容，除非它被移动到明确的 historical log 或加上充分 caveat。

## Decisions

### Decision 1: 新增“当前事实层”文档，而不是继续扩写 README

新增或整理以下文档：

- `docs/mainline_model_catalog.md`：主线模型和 baseline 目录。字段包括 `line_id`、研究问题、模型/registry key、config、入口命令、数据集/场景、split、target、metric profile、运行状态、对照组、结果账本引用和 caveat。
- `docs/experiment_protocols.md`：实验协议和参数口径表。字段包括 config family、formal/smoke/lowmem/debug/upper-bound 状态、seed、epoch、batch、lr、seq_len、num_pred、label space、GPS feature mode、selection split、输出目录和验证命令。
- `docs/result_claims_registry.md`：可引用结果账本。只记录源码可维护的摘要和本地产物路径引用，不提交真实产物。

替代方案是继续把内容写入 README 或 `docs/experiment_matrix.md`。不采用，因为 README 已经承担 quickstart，实验矩阵也已混合命令、协议和结论；继续扩写会放大读错概率。

### Decision 2: baseline 报告采用 current summary + historical log 分层

`README_REPRODUCE.md` 保持操作入口和当前推荐命令；`BASELINE_REPORT.md` 顶部必须给出 current summary，包含当前 Table III 本地 substitute 的正式口径：`beam_target_source=current`、`seq_len=1`、`num_pred=1`、`paper_distance_angle`、linear DBA 和 strict-validation/official/substitute caveat。

历史命令、`future` target ablation、test-as-validation upper-bound、dry-run/mock 和旧中间结果必须被放入 historical/appendix 段落，且每段显式标记不得作为当前正式结果。`results/reproduce_baseline.md` 可继续作为流水账，但必须在开头指向 current summary，并声明后续内容按时间记录、不可自动视为推荐口径。

### Decision 3: OpenSpec cleanup 优先清理 current spec 的内部冲突

对 `experiment-workflow` 做 delta：

- 修改“配置驱动实验”和“命令行覆盖配置”这类仍包含 KD active 场景的 requirement。
- 移除或改写旧 `student_no_kd`、Fusion KD、teacher/student 模型构建等 current wording。
- 保留拒绝旧 KD、旧 no-KD、旧 residual/Top8/Hist 的 guard 语义。

这样做比只在 README 加 caveat 更稳，因为 agent navigation 明确把 current specs 作为高优先级权威来源。

### Decision 4: 文档检查只做低风险静态约束

实现阶段可以扩展现有架构边界测试或增加轻量 helper，检查：

- 新增 mainline documentation spec 被 lifecycle inventory 分类为 `current`。
- README 和文档索引链接到主线模型目录、实验协议表和结果账本。
- current docs/specs 中对 `future target`、`test_as_validation`、`mock`、`smoke`、`upper-bound`、`legacy KD`、`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd`、Hist/Top8/residual/Raymobtime 等关键词的出现具有明确 historical/retired/supporting/upper-bound 限定。

检查不得读取真实 `dataset/`、`outputs/`、checkpoint 或训练日志；如果读取运行结果摘要，只能读取已跟踪文档。

## Risks / Trade-offs

- [Risk] 文档数量增加，读者反而不知道先看哪里。  
  Mitigation: README 只保留短索引，`docs/experiment_matrix.md` 指向主线目录和协议表；`docs/project_surface_inventory.md` 记录文档生命周期。

- [Risk] 结果账本被误认为提交了真实 metrics artifact。  
  Mitigation: 账本只保存摘要、口径和本地产物路径引用；明确真实 metrics、figures、cache 和 checkpoint 仍在 ignored `outputs/`。

- [Risk] 静态 wording 检查误杀历史说明。  
  Mitigation: 允许历史段落使用关键词，但要求附近出现 `历史`、`退役`、`upper-bound`、`mock`、`smoke`、`不可作为当前结果` 等限定词；先覆盖高风险词，再逐步扩展。

- [Risk] 清理 `experiment-workflow` 时误删仍有价值的 supporting helper 语义。  
  Mitigation: 只移除 active KD/teacher/student 运行要求；TopK metric、LOSO helper、artifact registry 和 migration guard 等 supporting 语义继续保留，并由 lifecycle inventory 标注。

- [Risk] baseline current summary 与历史流水账的数值不一致。  
  Mitigation: current summary 必须声明来源、日期、配置、target、selection split 和 claim status；历史日志保留原文但不作为当前事实覆盖 summary。

## Migration Plan

1. 新增主线文档和结果账本草案，先用现有 README、experiment matrix、配置 README、OpenSpec 和 baseline 报告填充。
2. 更新 README 和 `docs/experiment_matrix.md`，把长解释收敛为索引和当前推荐命令，复杂表格转到新文档。
3. 重排 `BASELINE_REPORT.md` 和 `results/reproduce_baseline.md`，让 current summary 位于开头，历史记录降级为 appendix/log。
4. 更新 `docs/project_surface_inventory.md`，登记新增文档和 OpenSpec capability lifecycle。
5. 清理 `experiment-workflow` 等 current specs 中的旧 active wording，并新增健康护栏检查。
6. 运行 `openspec validate strengthen-mainline-experiment-docs --strict`，再运行文档/架构相关 focused tests。

Rollback 策略：如果新增文档结构不合适，可以保留 specs 的治理要求，先撤回 README 索引和文档拆分改动；不会影响训练实现或本地产物。

## Open Questions

- `docs/result_claims_registry.md` 使用 Markdown 表还是 YAML/JSON sidecar 更适合长期维护？初版建议 Markdown 表，后续如需机器校验再增加结构化 sidecar。
- `BASELINE_REPORT.md` 是否保留所有历史命令全文，还是把过长流水账移动到 `results/reproduce_baseline.md` 并只保留摘要？初版建议保留但加 clear appendix 标识。
- 对 wording 静态检查的严格度应一次到位还是分阶段增强？初版建议先覆盖最高风险词，避免误杀大量历史上下文。
