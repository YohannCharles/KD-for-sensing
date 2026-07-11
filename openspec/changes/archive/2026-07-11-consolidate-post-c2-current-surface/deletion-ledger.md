## Baseline

口径：2026-07-11，全部统计仅使用 `git ls-files`、`git status --short`、`git diff`、`pyproject.toml` 和 tracked source/docs/tests/OpenSpec；未读取 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard 或 `All_models/`。

| Surface | Tracked files | Lines |
| --- | ---: | ---: |
| `src/kd_sensing` | 282 | 84,294 |
| `tests` | 72 | 21,598 |
| `scripts` | 28 | 8,362 |
| `openspec/specs` | 111 | 20,044 |

基线工作树：

```text
 M scripts/launch_h5_p1_temporal_models_v1.py
?? openspec/changes/consolidate-post-c2-current-surface/
```

用户 H5/P1 launcher diff：`+3/-0`；增加 `--auto-resume` parser flag，并在 `plan_jobs()` 中向训练命令追加 `--auto-resume`。基线 worktree blob 为 `114c915f61bc2095ef10caf27c1ef8c5905c8551`。每个删除 wave 后都必须重新检查该 diff，不能格式化或覆盖它。

基线 console scripts 共 13 个：

1. `kd-sensing-train`
2. `kd-sensing-evaluate`
3. `kd-sensing-preprocess`
4. `kd-sensing-runs`
5. `kd-sensing-research-dashboard`
6. `kd-sensing-research-preview`
7. `kd-sensing-clean-runtime-artifacts`
8. `kd-sensing-organize-runtime-outputs`
9. `kd-sensing-paper-export`
10. `kd-sensing-eval-u-mask-matrix`
11. `kd-sensing-mmw-town-gps-v2`
12. `kd-sensing-inspect-mmw-physics`
13. `kd-sensing-project-surface-doctor`

## Deletion Groups

| Wave | 删除组 | Tracked source consumer | Config / CLI / spec / claim consumer | 替代 owner | Focused validation | Rollback |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | temporal v1、S1-S4 wrappers/branches、overnight launcher | S1-S4 wrappers 只相互导入 H5/P1 script 私有 helper；提前实现只由专属 test/wrapper 消费；overnight launcher 无 package consumer | `add-temporal-window-missing` 未完成 S1-S4 tasks、script inventory 和专属 tests；final C2 只消费 overnight summary parser | H5/P1 parameterized launcher/eval/summary、temporal difficulty owner、retained overnight summary | temporal/H5-P1/U-Mask/final-C2 tests、compile、architecture | 恢复 Wave 2 文件及同波 docs/tests/spec；不恢复 parallel wrapper suite |
| 3 | research dashboard/preview/harvester、旧 Scene31 summary | Dashboard/preview 只由对应 CLI 相互消费；harvester modules 只服务 dashboard；`scene31_summary/` 只服务旧 baseline/next-round scripts/tests | 3 个 public CLI 中的 dashboard/preview；retired claim/Scene31 specs 和旧 docs | `kd-sensing-runs`、人工 claim registry、formal protocols、paper export、`scene31_34_final_analysis/` | run-index/cleanup/paper-export/Scene31-34/AMR/AMBER tests | 恢复 Wave 3 owner 与同波 CLI/docs/tests；不新增 HTML 或 shared-summary facade |
| 4 | standalone training-I/O profile/recommendation、legacy LOSO、窄 orphan、architecture summary renderer | profile/recommendation 仅被五个 training-I/O tests 消费；`data.loso` 无 current source consumer；orphan facade 逐项以 tracked import 复核；architecture summary 的 instance/startup 路径有 current consumer | retired/speculative specs 和专属 tests；无保留 public CLI | dataset/cache/trainer/evaluator、`data/mmw/protocol.py`、`data/target_shot_splits.py`、真实 MMW owner、instance/startup architecture summary | training-I/O、MMW、U-Mask、AMR/AMBER、registry/objective/physics tests | 逐 owner 恢复；发现 current consumer 时标记 `retained-with-evidence`，不以 stub 替代 |
| 5 | GPS-query/predictive JEPA、Scenario-D/CxD 专属 difficulty | Query/predictive branches 由 JEPA builder、batch/objective/metadata 和专属 tests 消费；current MMW config 只消费 `jepa_context_image` mean pooling/checkpoint extraction | retired query configs/specs/docs/tests；无 current claim | JEPA pretraining owner、mean context encoder/checkpoint extraction、generic GPS/image difficulty 和 missing stress | JEPA/difficulty/config/registry/optimizer/objective/MMW/architecture tests | 恢复整组 Wave 5 分支；不得把 retired pooler静默映射到 mean |
| 6 | geometry prior 与 safe reranker | modular construction/forward、batch、objective、metadata、architecture summary 相互消费，外部只剩专属 tests/retired configs | retired geometry/rerank specs、tests 和文档；无 current claim/config | ordinary modular/U-Mask fusion/head、MMW geometry preparation、physics scoring、generic Top-K/DBA metrics | modular/registry/objective/difficulty/physics/MMW/architecture tests | 原子恢复 Wave 6 model+forward+batch+loss+tests；不恢复兼容 alias |
| 7 | 24 个独立 tombstone specs、文档与治理镜像 | 无 runtime consumer；防回流由集中 guard 承接 | lifecycle inventory、current specs、README/docs/architecture mirrors | `retired-route-summary`、pyproject、inventory、current owner specs | OpenSpec strict、architecture/config/CLI/retired-route tests | 恢复本 wave docs/tests/spec delta；apply 阶段不手删 `openspec/specs/` |
| 8 | project surface doctor product | CLI/doctor tests/Make/docs 消费；唯一独立价值是 secret/system-config/dangerous-shell guard | `kd-sensing-project-surface-doctor`、doctor spec/docs/test | 小型标准库安全 fixture、architecture/CLI/config/compile 原生检查 | doctor 最后只读扫描；之后 architecture/CLI/config/run-index/cleanup/compile | 恢复 doctor owner+CLI+docs/tests；不保留 alias/stub/report schema |

## Wave 7 Capability Audit

以下固定 24 项已逐项比较 current spec 与本 change delta 的 requirement 标题；每个 current requirement 均有同名 `REMOVED` delta。源码、CLI、config、claim 和专属 tests 已在对应删除 wave 清除，拒绝与历史迁移统一由 `retired-route-summary`、ordinary unknown-name 行为、`config/migration_guards.py` 和 `tests/test_retired_routes.py` 按实际入口承接，不再保留逐 capability guard。

| Capability | Removed requirements | Current consumer / independent guard | Consolidated boundary |
| --- | ---: | --- | --- |
| `beam-distribution-shift-diagnostics` | 5 | 无 | retired route/docs context |
| `beambench-baseline-reproduction` | 1 | 无 | config/registry unknown guard |
| `bev-fusion-2604-reproduction` | 1 | 无 | config/registry unknown guard |
| `cxd-phase-transition-analysis` | 10 | 无 | retired route/docs context |
| `dataset-reproducibility-audit` | 3 | 无 | retained dataset contracts/tests |
| `geometry-prior-beam-fusion` | 11 | 无 | ordinary component unknown guard |
| `gps-query-effectiveness-visualization` | 9 | 无 | retired query token guard |
| `gps-query-jepa-pooling` | 5 | 无 | retired pooler fail-fast |
| `jepa-gps-shortcut-benchmark` | 1 | 无 | retired CLI/config guard |
| `jepa-visual-analysis-suite` | 1 | 无 | retired CLI/config guard |
| `jepa-visual-architecture-sweep` | 1 | 无 | retired config/path guard |
| `modality-visual-diagnostics` | 1 | 无 | retained generic modality metadata |
| `predictive-jepa-robustness` | 18 | 无 | retired config/pooler fail-fast |
| `rbma-prototype-kd-missing-workflow` | 1 | 无 | retired KD/config guard |
| `real-perturbation-forward-evaluation` | 6 | 无 | retained generic difficulty tests |
| `safe-residual-beam-rerank-fusion` | 6 | 无 | ordinary component unknown guard |
| `scenario-d-image-observability-benchmark` | 10 | 无 | retired condition/token guard |
| `scene31-baseline-pack` | 6 | 无 | retained Scene31-34 final owner |
| `scene31-next-round-experiment-workflow` | 24 | 无 | retained Scene31-34 final owner |
| `scenes31-34-subset-reliability-validation` | 2 | 无 | retained final evidence owner |
| `tii-vlrg-transformer-reproduction` | 4 | 无 | config/registry unknown guard |
| `training-throughput-optimization` | 1 | 无 | retained run metadata/tests |
| `vision-position-baseline-suite` | 8 | 无 | config/registry unknown guard |
| `wcl2025-robust-missing-modality-reproduction` | 5 | 无 | retired CLI/config guard |

## Protected Paths

- Final C2 / U-Mask：`src/kd_sensing/models/u_mask_beam_jepa.py`、`src/kd_sensing/models/reliability_biased_missing_attention.py`、U-Mask loss/objective、`configs/fusion/u_mask_beam_jepa*.yaml`、final C2 scripts/tests/specs；保护 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router`、`reliability_biased_missing_attention`、beam prototype alignment、full-to-partial teacher stabilization 和 pattern metrics。
- MMW / CSI / physics：`src/kd_sensing/data/mmw/`、`src/kd_sensing/data/datasets/mmw_geometry.py`、`src/kd_sensing/engine/mmw_town_gps_v2.py`、`src/kd_sensing/models/physics/`、CSI owners/configs、MMW/CSI/physics focused tests。
- Scene31-34 final evidence：`src/kd_sensing/diagnostics/scene31_34_final_analysis/`、`src/kd_sensing/diagnostics/scene31_eval_resolution.py`、Scene31-34 generator/runner、missing statistics/stress、paper evidence inputs。
- Runtime safety：run index、`run_index_resources.py` 的 PID discovery/matching、memory/GPU snapshot 和稳定 `resources` schema；runtime cleanup manifest、tracked-file protection、active-run protection 和 explicit confirmation。
- Temporal supporting：H5/P1 launcher/eval/summary、temporal difficulty/window aliases、`tests/test_temporal_window_missing.py`、`tests/test_h5_p1_temporal_matrix_v1.py`，以及 `src/kd_sensing/models/rmbp_mm.py` channel-attention core。
- Split/summary supporting：`src/kd_sensing/data/target_shot_splits.py` 及 MMW protocol consumer；`models/architecture_summary.py` 的 instance parameter/component/trainability schema、training startup artifact 和 Scene31-34 profile consumer。

## Retained-With-Evidence

实现中发现候选存在 current consumer 时在此追加调用路径、owner、focused test 和未来删除触发条件；不得以“以后可能使用”为理由。

| Candidate | Current call path | Owner / focused validation | Future deletion trigger |
| --- | --- | --- | --- |
| `src/kd_sensing/eval/missing_patterns.py` | `trainer_runtime_helpers`、U-Mask eval matrix/CLI、difficulty presets、missing-modality statistics/stress 通过该兼容导入面调用 `utils.missing_patterns` | `utils/missing_patterns.py`; `tests/test_u_mask_beam_jepa_eval_matrix.py`、`tests/test_missing_modality_statistics.py`、`tests/test_missing_modality_stress.py` | 所有 current consumer 统一迁移到单一 owner，且 OpenSpec `modality-contracts` 不再要求该导入面 |

## Final Verification

分波删除的 tracked 文件数：Wave 1 active-change 收口 2 个，Wave 2 temporal/launcher 8 个，Wave 3 dashboard/harvester/旧 Scene31 21 个，Wave 4 training-I/O/LOSO/orphan 9 个，Wave 5 query/downstream/doc 3 个，Wave 6 geometry/reranker 3 个，Wave 7 不物理删除 current specs，Wave 8 doctor product 3 个；合计 49 个。共享 owner 的行级裁剪计入整体净删，不按 wave 重复分摊。

| Surface | Baseline | Final apply tree | Delta |
| --- | ---: | ---: | ---: |
| `src/kd_sensing` files | 282 | 249 | -33 |
| `src/kd_sensing` lines | 84,294 | 66,425 | -17,869 |
| `tests` files | 72 | 66 | -6 |
| `tests` lines | 21,598 | 18,366 | -3,232 |
| `scripts` files | 28 | 21 | -7 |
| `scripts` lines | 8,362 | 7,201 | -1,161 |
| physical current specs | 111 | 111 | 0 during apply |
| effective post-archive specs | 111 | 81 | -30 after independent archive |
| public console scripts | 13 | 10 | -3 |

验证结果：OpenSpec strict 114/114；focused architecture/config/CLI/retired/run-index/cleanup 147 passed；compile 32 tracked CLI/script Python files；full pytest 958 passed、126 个既有 PyTorch warnings。最终 `git diff --check` 通过，无 dataset/output/log/cache/checkpoint/`All_models` 状态项。用户 H5/P1 launcher 保持 `+3/-0`，worktree blob 仍为 `114c915f61bc2095ef10caf27c1ef8c5905c8551`。

Known caveat：30 个全量 `REMOVED` delta 在 apply 阶段仍对应 111 个 physical current specs；必须通过独立 archive workflow 合并后才得到 effective 81。三个 changes 只标记 ready-to-archive，本 change 不执行 archive。各 wave rollback 继续使用上表的原子 owner 边界；不得只恢复 wrapper、alias、doctor schema 或静默 fallback。
