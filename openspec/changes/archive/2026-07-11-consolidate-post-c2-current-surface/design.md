## Context

当前仓库的运行源码约 84,294 行，主要集中在 `diagnostics`、`data`、`engine` 和 `models`。代码体量并非单一巨型模块造成，而是三类表面叠加：

1. 已退役研究线仍保留 current specs、专属实现、测试或文档入口；
2. 一次性实验逐步长成 launcher/eval/summary 三件套和独立诊断产品；
3. inventory、maintainer index、architecture tests、surface doctor 和 OpenSpec 多次镜像同一事实。

审计得到以下基线：

- `src/kd_sensing`: 282 个 Python 文件，84,294 行；
- `tests`: 72 个文件，21,598 行；
- `scripts`: 28 个文件，8,362 行；
- current OpenSpec: 111 个 capability 目录、20,044 行，其中 inventory 分类为 71 `current`、13 `supporting` 和 27 `retired-tombstone`；
- OpenSpec archive: 1,813 个文件、约 80,389 行，只是历史记录，不属于运行架构优化范围；
- `add-temporal-window-missing` 仍为 17/25，S1-S4 tasks 未完成但实现已提前存在；
- `align-paper-baselines-window2` 显示 10/10，但其 tasks 声称存在的部分 RMBP/WCL surface 实际不存在；
- 工作树只有用户对 `scripts/launch_h5_p1_temporal_models_v1.py` 的 3 行 `--auto-resume` 改动，本 change 必须原样保留。

当前目标架构不是重新分层，而是保留已经存在的最短主干：

```text
package CLI
    -> config normalization/validation
    -> data + difficulty
    -> engine train/evaluate/preprocess
    -> models + losses
    -> ignored runtime artifacts

supporting:
    run index -> runtime cleanup
    claim docs -> paper export
    MMW/CSI -> current supporting workflow
```

任何不在该依赖流中、没有 current public contract、没有 current consumer、也不提供独立安全边界的实现，默认退出源码。

## Goals / Non-Goals

**Goals:**

- 将 README、OpenSpec、inventory、pyproject 和真实源码统一到 final C2 / U-MaskBeamJEPA 主线与 MMW/CSI supporting 边界。
- 删除高置信 retired/zero-consumer runtime，目标净减不少于 9,000 行 `src/kd_sensing` 源码；低于该阈值时必须在 implementation notes 中逐项解释保留原因。
- 将 post-archive current specs 从 111 个收敛到预期 81 个、最多 84 个；27 个 `retired-tombstone` 中经审计折叠 24 个，将仍有 current consumer 的 `target-shot-domain-splitting`、`model-architecture-summary` 与 `local-missing-modality-baselines` 改为 supporting。81 到 84 的余量只允许用于 implementation 发现并记录 `retained-with-evidence` 的真实消费者，不允许保留“以后可能用”。
- 从 public CLI 删除 surface doctor、research dashboard 和 research preview，同时保持核心 train/evaluate/preprocess、run index、runtime cleanup、paper export、U-Mask eval 与 MMW CLI 稳定。
- 删除 S1-S4 未完成表面和被 H5/P1 覆盖的 temporal v1 脚本，不建立新的通用 launcher framework。
- 让架构 guardrail 检查真实结构和安全边界，而不是维护完整 deleted-path、文案和 lifecycle 镜像。
- 每个 wave 独立验证、独立计数、可单独停止，不把失败累积到最后一次全量回归。

**Non-Goals:**

- 不修改 final C2 / U-MaskBeamJEPA 已有 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router` 分支及其训练数学语义。
- 不删除 Scene31-34 final analysis、missing-modality statistics/stress、current claim docs 或 paper export；这些表面只有在论文证据冻结后的独立 change 中才能退役。
- 不删除或重构 MMW/CSI/physics-informed MMW 的 current owner、配置、CLI 和 focused tests。
- 不收缩 canonical YAML；审计没有发现完全重复配置，为省 YAML 引入继承或 recipe engine 不划算。
- 不删除 OpenSpec archive，也不移动历史代码到新的 `legacy/`、`archive_runtime/` 或兼容 package。
- 不改 dataset split、beam label/label-space、metric schema、checkpoint schema、run metadata、默认输出路径或本地产物边界。
- 不修改、移动或清理 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard、`All_models/`。

## Decisions

### Decision 1: Wave 0 先关闭 active-change 冲突

任何源码删除前先处理两个 active change：

- `align-paper-baselines-window2`: 该 change 已 10/10 complete，但其 WCL/local-baseline delta 与真实文件及 post-C2 lifecycle 冲突。独立 closeout MUST 先把 artifacts 收缩到真实 AMBER/AMR + window 2/1 内容，或明确 abandon 整个 change；MUST NOT 为满足旧 tasks 恢复 WCL CLI/config/spec。H5/P1 正在消费的 `rmbp_mm` channel-attention core 继续作为 temporal supporting component 保留。完成后只标记为 ready-to-archive，归档仍走独立 archive 流程。
- `add-temporal-window-missing`: 保留 history/prediction alias、temporal difficulty、H5/P1 matrix 和用户 `--auto-resume` 改动；MUST 先修改该 change 的 proposal/design/spec/tasks，把 S1-S4 以及已被 focused tests/H5/P1 覆盖的旧 temporal check/launch/summary 从 scope 移除并记录未来新 change 触发条件，然后删除相应 model/loss/scripts/tests 的提前实现。

替代方案是让三个 change 并行实施。该方案会让同一 temporal/model/script 文件同时受两个 change 控制，因此不采用。

Wave 0 停止条件：active artifacts、工作树和 deletion ledger 无未记录重叠；架构测试不再因 S1-S4 wrapper 失败。

### Decision 2: 以 capability 删除代替文件重排

本 change 不追求更少目录或平均文件长度，而按以下证据删除完整能力：

| 证据等级 | 条件 | 动作 |
| --- | --- | --- |
| A | inventory 已 retired，且无 current config/CLI/src consumer | 直接删除实现、专属测试和独立 tombstone |
| B | 无 src/script/config consumer，只有专属测试或 future spec | 删除实现和测试；历史语义留 docs/archive |
| C | 有 current owner，但部分 branch/render/resource path 零 caller | 只删除不可达分支，保留 owner/public schema |
| D | current claim、MMW/CSI、U-Mask、cleanup safety 消费 | protected，不进入本 change |

删除判定使用 tracked-only 引用、pyproject、current configs、current specs、inventory 和 focused tests。`outputs/` 或历史 archive 中的路径不构成 current consumer。

替代方案是把 orphan 合并到大 owner。零消费者代码合并后仍需维护，因此不采用。

### Decision 3: 运行源码按四组删除

#### Group A: 完整退出的 retired/zero-consumer owner

- `src/kd_sensing/diagnostics/scene31_summary/`；
- `src/kd_sensing/diagnostics/gps_query_evidence.py`；
- `src/kd_sensing/data/loso.py`；
- `src/kd_sensing/engine/training_io_profile.py`；
- `src/kd_sensing/engine/throughput_recommendations.py`；
- `src/kd_sensing/utils/geometry.py`；
- `src/kd_sensing/utils/checkpoint_resolver.py`；
- `src/kd_sensing/models/mmw_town_gps_v2.py`，保留真实 public owner `engine/mmw_town_gps_v2.py`；
- `src/kd_sensing/baselines/amber_lite.py` 中仅被专属测试消费的 facade；
- `src/kd_sensing/evaluation/physics_metrics.py`；
- `src/kd_sensing/eval/missing_patterns.py` re-export facade。

删除 training-I/O profiler/recommendation 时，五个 `tests/test_training_io_*` 文件不得按文件名整删：只移除 profiler/recommendation imports 与专属断言，继续保留其中覆盖 dataset、cache policy、trainer/evaluator 和 `engine.run_metadata.throughput_run_metadata()` 的 current tests。`throughput_run_metadata()` 本身位于 current run-metadata owner，明确保留。

#### Group B: 从 retained owner 中剪除 retired branch

- `jepa_downstream.py` 中 GPS-query、predictive、hybrid、query-weighted 等非 mean pooler；
- `jepa_downstream_helpers.py` 中只服务上述 pooler 的配置/shape helper；
- modular/config/difficulty 中只服务 GPS-query 的条件分支；
- difficulty presets/operators 中只服务 Scenario-D、P0-P5 predictive、CxD/GPS-query advantage、visual-hard-negative 与旧 shortcut benchmark/cache compatibility 的分支；保留通用 GPS delay/image degradation/missing-stress operators；
- `models/geometry_prior.py`、对应 diagnostic、modular forward/batch/run metadata 分支与专属测试；
- `models/architecture_summary.py` 中零 caller 的 config/sweep/render/artifact 路径，保留 startup 参数量和 trainability summary 所需最小函数。

保留 `gps-conditioned-jepa-pretraining` 和当前 MMW config 使用的 `jepa_context_image pooling: mean`。实现必须用 current config characterization test 证明 mean 路径未被误删。

#### Group C: 删除治理/展示产品面

- `project_surface_doctor.py`、其 CLI 和专属测试；
- `research_claim_harvester*`、`research_run_preview.py`、两个 CLI 和专属 tests；
- `research_claim_harvester_base.py`、`research_claim_harvester_collectors.py`、`research_claim_harvester_gate.py`、`research_claim_harvester_writers.py` 及其 dashboard/CLI 包装。

保留 `kd-sensing-runs` artifact/status scan、runtime cleanup/organize、人工 claim registry 和 paper export。Research dashboard 的 HTML/JSON 输出不迁移到新工具；Git 历史保留旧实现。

#### Group D: 明确保留

- `scene31_34_final_analysis/`；
- `missing_modality_statistics.py` 与 `missing_modality_stress.py`；
- runtime cleanup 全部 manifest/protection/confirm 行为；
- `run_index_resources.py` 的 process discovery/matching、memory/GPU snapshot 与稳定顶层 schema；cleanup 默认依赖该进程信号保护 active run；
- `data/mmw/protocol.py` 与 `data/target_shot_splits.py` 的 MMW split、leakage 和 artifact supporting contract；
- `models/architecture_summary.py` 中被 startup diagnostics、U-Mask/AMR/AMBER 和 Scene31-34 profile 消费的 instance/trainability summary；
- H5/P1 temporal matrix 及其 `rmbp_mm` supporting core；
- final C2、U-Mask、MMW、CSI、physics-informed MMW；
- `models/reliability_biased_missing_attention.py` 与 U-Mask 内嵌 RBMA、beam prototype alignment、full-to-partial teacher stabilization、pattern-balanced metrics；退役的只是旧独立 RBMA/prototype-KD sweep/runbook；
- canonical config 与 current paper/claim evidence。

### Decision 4: Public CLI 从 13 个收缩到 10 个

保留：

1. `kd-sensing-train`
2. `kd-sensing-evaluate`
3. `kd-sensing-preprocess`
4. `kd-sensing-runs`
5. `kd-sensing-clean-runtime-artifacts`
6. `kd-sensing-organize-runtime-outputs`
7. `kd-sensing-paper-export`
8. `kd-sensing-eval-u-mask-matrix`
9. `kd-sensing-mmw-town-gps-v2`
10. `kd-sensing-inspect-mmw-physics`

删除：

- `kd-sensing-project-surface-doctor`
- `kd-sensing-research-dashboard`
- `kd-sensing-research-preview`

删除的命令不提供 deprecation trampoline。内部调用方改为直接使用保留 owner，文档删除推荐命令，CLI help smoke 动态从 pyproject 的 current 列表生成。

### Decision 5: Script 收敛采用删除，不新建 runner framework

删除：

- `launch_temporal_router_s1_s4_v1.py`
- `eval_temporal_router_s1_s4_matrix_v1.py`
- `summarize_temporal_router_s1_s4_v1.py`
- `check_temporal_window_missing.py`
- `launch_temporal_missing_v1.py`
- `summarize_temporal_missing_v1.py`
- `launch_overnight_branch_router_v2.py`

保留用户正在修改的 H5/P1 launcher/eval/summary、final C2 launcher/summary、PCPG/BPRR current helper、Scene31-34 protected generator/runner、overnight summary（仍被 final C2 summary 直接导入）、MMW helper和 compile verification。`overnight-branch-router-v2` current spec 同步删除 launcher requirement，并将 summary 与 focused test 收缩为 supporting parser contract。

H5/P1 已支持 method、root、cache 和 output 参数，不再通过 `sys.path.insert`、导入脚本私有函数或改写模块全局变量派生第二套 suite。

### Decision 6: 24 个 tombstone 集中折叠，3 个误分类 owner 改为 supporting

`docs/project_surface_inventory.md` 中 27 个 `retired-tombstone` 逐项检查 source consumer、current spec 与独立 guard。`target-shot-domain-splitting` 被 MMW protocol 直接复用，`model-architecture-summary` 被 startup diagnostics、U-Mask/AMR/AMBER tests 与 Scene31-34 profile 消费，`local-missing-modality-baselines` 中的 AMR-lite 被 protected Scene31-34 runner/config消费，因此三者改为 `supporting`；其余 24 个从 `openspec/specs/` current set 删除。`retired-route-summary` 保留：

- 旧 capability 名称和代表性 CLI/config/module token；
- 普通 unknown-name 或集中 migration guard 的拒绝语义；
- 通用 Top-K、circular metric、label-space、JEPA mean pool 等可由 current owner 继续复用的边界；
- 历史细节通过 git 与 dated OpenSpec archive 查询。

折叠清单固定为：

1. `beam-distribution-shift-diagnostics`
2. `beambench-baseline-reproduction`
3. `bev-fusion-2604-reproduction`
4. `cxd-phase-transition-analysis`
5. `dataset-reproducibility-audit`
6. `geometry-prior-beam-fusion`
7. `gps-query-effectiveness-visualization`
8. `gps-query-jepa-pooling`
9. `jepa-gps-shortcut-benchmark`
10. `jepa-visual-analysis-suite`
11. `jepa-visual-architecture-sweep`
12. `modality-visual-diagnostics`
13. `predictive-jepa-robustness`
14. `rbma-prototype-kd-missing-workflow`
15. `real-perturbation-forward-evaluation`
16. `safe-residual-beam-rerank-fusion`
17. `scenario-d-image-observability-benchmark`
18. `scene31-baseline-pack`
19. `scene31-next-round-experiment-workflow`
20. `scenes31-34-subset-reliability-validation`
21. `tii-vlrg-transformer-reproduction`
22. `training-throughput-optimization`
23. `vision-position-baseline-suite`
24. `wcl2025-robust-missing-modality-reproduction`

不在该清单中的 spec 不得因目录名或旧 inventory 分类被顺手删除。

不为每条旧路线新增专属 test 或 10 行小墓碑。一个参数化 guard 验证旧 CLI/config/module 不回流，并扫描 current README/inventory/agent docs 的推荐语境。

本 change 为 24 项提供逐 requirement 的 `REMOVED` delta；apply 阶段不得直接手改或删除 `openspec/specs/<capability>`，独立 archive workflow 才将空 capability 从 current specs 移除。实现阶段更新 target lifecycle inventory 与 source/docs/tests，并在 deletion ledger 中证明每一项无 current consumer/独立 guard。`target-shot-domain-splitting` 保留全部 split contract；`model-architecture-summary` 只删除 standalone CLI、candidate sweep、renderer/config-preflight 分支，保留 instance/startup schema；`local-missing-modality-baselines` 只保留 AMR-lite supporting requirement，其余重复要求删除。

### Decision 7: Mainline wording 只保留一个现行口径

`mainline-experiment-documentation` 中将 Image+GPS JEPA、Vision-Position、BeamBench、JEPA visual/GPS shortcut 描述为 current 的旧 requirements 删除。现行口径为：

- 默认主线：final C2 / U-MaskBeamJEPA missing-modality beam prediction；
- supporting：MMW/CSI、保留 JEPA mean-pooling/pretraining 组件、current paper evidence；
- historical/retired：其余 post-C2 已退出路线；
- current capability 不等于 claim 已 verified，claim registry 继续记录 pending/smoke/caveat。

### Decision 8: Surface doctor 用小型 guard 替代

删除 1,447 行 doctor owner，而不是继续修补 substring heuristic。保留以下可执行检查：

- pyproject console scripts 与 CLI help smoke 一致；
- tracked runtime artifacts 不进入源码；
- retired token 不作为 current command/config/module 回流；
- 系统配置/凭证文件不包含训练、清理或启动命令；
- current docs 的 validation target 存在；
- active complete change 有 archive 或 deferral 记录。

这些检查放在现有 architecture/focused tests 中，使用标准库 `tomllib`、`ast`、`pathlib` 和 `subprocess`。不创建新的 doctor、lint framework 或 JSON report schema。

### Decision 9: Maintainer index 只保留不可推导路由

保留 `docs/maintainer_context_index.yaml`，但只包含：

- route id；
- scoped context path；
- authority paths；
- owner roots；
- focused validation；
- retired-route guard 引用。

删除 CLI mirror、script inventory、hotspot budget、remediation waves、完整统计基线和重复 lifecycle metadata。CLI 以 pyproject 为权威，scripts/configs 以 inventory 为权威，需求以 current specs 为权威。

替代方案是完全删除 index。AGENTS 与 scoped context 当前仍依赖其路由价值，因此保留最小版本。

### Decision 10: 以净删量和行为验证共同验收

净删量只作为防止“删除一个 wrapper、增加一个 framework”的辅助门：

- `src/kd_sensing` 净减目标不少于 9,000 行；
- current specs 数量目标不超过 84；
- public CLI 固定为 10；
- 不新增第三方依赖；
- 不新增 `legacy`、compat、generic launcher、doctor replacement package；
- 所有 protected behavior focused tests 通过。

若某候选因真实 current consumer 保留，implementation notes 必须给出调用路径、owner、验证命令和未来删除触发条件。不能用“可能以后需要”作为保留理由。

## Risks / Trade-offs

- [Risk] GPS-query 删除误伤 MMW JEPA mean pooling。→ Mitigation：先生成 class/function/config consumer map，只删除非 mean 分支；运行 JEPA config characterization 与 MMW focused tests。
- [Risk] Geometry prior 分支散布在 modular forward、batch、metadata 和 tests。→ Mitigation：按 registry/config -> model -> forward -> batch -> metadata -> tests 的逆依赖顺序删除，并运行 modular sequence focused tests。
- [Risk] 删除 research dashboard 后失去自动 HTML 概览。→ Mitigation：不迁移展示功能；保留 run index、claim registry 和 paper export 这三个实际证据 owner。
- [Risk] 删除 doctor 后遗漏安全检查。→ Mitigation：把唯一独立的 secret/system-config/shell-runner guard 保留为小型参数化测试，并证明它能对真实危险 fixture 失败。
- [Risk] current tombstone 删除后旧路线不易查询。→ Mitigation：集中 summary 保存名称和拒绝点，dated archive 与 git 保存完整历史。
- [Risk] active temporal change 与用户 dirty launcher 重叠。→ Mitigation：Wave 0 记录 blob/diff，只删除 S1-S4 和 superseded 文件，不修改用户 3 行 `--auto-resume`。
- [Risk] 大量删除导致失败难定位。→ Mitigation：每个 group 分波提交/验证，上一波未通过时停止；不在失败状态继续下一波。
- [Risk] 代码行目标驱动误删。→ Mitigation：protected inventory 与 focused behavior 优先于净删指标；目标只约束不得新增替代复杂度。

## Migration Plan

1. 记录 `git status --short`、`openspec list --json`、当前文件/行数、CLI 列表和 baseline validation；保存用户 dirty diff 摘要。
2. 修正两个 active change 的 artifacts 与实际实现状态；S1-S4 scope defer，align-paper false-complete 收口但不在本 change 内归档。
3. 更新 change delta、README、inventory 和主线 wording，建立 protected/deletion ledger。
4. 删除 temporal/overnight scripts 与 research dashboard/preview CLI，运行 CLI/compile/architecture focused validation；surface doctor 此时暂留作过渡检查。
5. 删除 research harvester/preview、旧 Scene31 summary 等隔离 diagnostics owner，运行 claim docs、run index、cleanup、paper export 和 Scene31-34 focused validation。
6. 删除 GPS-query/Scenario-D、geometry、LOSO、training-I/O profiler 与其它 orphan runtime，按领域运行 focused tests。
7. 验证 24 个 all-requirements `REMOVED` delta、更新 target lifecycle inventory、缩减 maintainer index 和 architecture tests；current spec 物理删除留给独立 archive。
8. 将 doctor 唯一独立的 secret/system-config/dangerous-shell 检查迁入轻量结构测试，再删除 doctor owner/CLI/tests 和最后的 current reference。
9. 复核净删量、十个 public CLI、protected paths、tracked artifact 边界和 stale references。
10. 运行 full verification；实现完成后由独立 archive 流程归档 active changes。

Rollback 以 wave 为单位：恢复当前 wave 删除文件及其同波 docs/tests/spec 引用，不恢复旧入口的兼容 wrapper。若某 owner 被证明有 current consumer，将其恢复并在 ledger 标记 `retained-with-evidence`，其它已通过 wave 不回滚。

## Open Questions

无阻塞设计问题。以下事项已采用保守默认值：

- Scene31-34 final analysis、missing-modality statistics/stress 暂保留；
- runtime cleanup、run index artifact scan、paper export 暂保留；
- research dashboard/preview 与 surface doctor 删除；
- H5/P1 temporal matrix 保留，S1-S4 defer；
- canonical configs 与 OpenSpec archive 不做体量清理。
