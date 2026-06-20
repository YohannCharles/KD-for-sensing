## Context

本 change 是一次表面积收缩，不是新模型或新实验路线。它建立在几个既有事实上：

1. `retire-legacy-model-registry-surface` 已经把 legacy registry 名称列为优先退役项，并正在处理配置迁移、removed guard、文档和测试收口。本 change 不重复实现它，而是把它作为后续退役波次的前置条件。
2. MMW Town GPS v2 仍是 current formal diagnostic，但它旁边的三个 `scripts/mmw/visualize_gps_*` 脚本已经和 package plotter/comparison CLI 形成重叠，且长期维护入口过多。
3. AMR-Net_gps_image 和 JEPA-MSAC 当前主要是 blocked/mock/local-substitute 账本入口。它们有 current CLI、config、tests 和 claim 行，但没有可引用真实结果；继续保留 current runner 会让主线目录看起来比实际研究面更大。
4. 若干 shell orchestration 脚本只是本地实验 convenience wrapper，默认写 ignored outputs/logs。它们适合作为历史命令或 runtime artifact cleanup 背景，不适合继续占据 current scripts allowlist。

这四点都不是“删掉历史说明”。历史 caveat、blocked 原因和 migration guard 可以保留在 OpenSpec tombstone、inventory 或 result registry 的历史段落中；要退役的是 current 可运行入口、实体配置、console script、allowlist 和 focused tests。

## Goals / Non-Goals

**Goals:**

- 给出四个退役点的实施顺序、边界、迁移目标和验证命令。
- 让 current 入口只保留实际推荐维护的 package CLI、配置和诊断 workflow。
- 保留历史/blocked 背景，但从 mainline catalog、claim registry current 行、pyproject entry points 和 script allowlist 中移除。
- 增加测试和文档 guard，防止旧 CLI、script、config、module path 和 claim 行回流。
- 不读取真实 `dataset/`，不训练，不生成 checkpoint/cache/log，不删除本地 `outputs/` 或 `logs/`。

**Non-Goals:**

- 不退役 `mmw_town_gps_v2` runner、plotter、comparison 这条 formal diagnostic 主线。
- 不退役 CSI hardening matrix shell runner。
- 不退役 BeamBench/Arnold22 Camera AE+GPS Direct、本地 Vision-Position baseline suite、JEPA visual analysis 或 GPS shortcut benchmark。
- 不在本 change 中重新设计模型 registry 迁移；registry 代码收口由 `retire-legacy-model-registry-surface` 完成，本 change 只检查其完成状态并更新后续表面积。
- 不清理本地历史输出；如用户之后要清 outputs/logs，应走 runtime artifact cleanup manifest。

## Decisions

### Decision 1: 以 `retire-legacy-model-registry-surface` 为前置门

实施本 change 前先确认 active registry change 完成以下状态：

- legacy whole-model/alias/feature-extractor 名称不再出现在 current registry list。
- `docs/model_architecture_inventory.md` 不再把 legacy 名称列为 current model/core/head/encoder。
- root/canonical configs 已经走 `model.primary.type: modular_sequence` 或明确保留的 paper/workflow 例外。
- `openspec validate retire-legacy-model-registry-surface --strict` 与 focused tests 有记录。

本 change 的任务中只保留“检查并记录前置状态”的步骤，不重新修改同一批 registry 代码。这样能避免两个 active change 同时编辑 `src/kd_sensing/models/*`、`docs/model_architecture_inventory.md` 和 registry tests。

Alternative: 把 registry 收口也并入本 change。拒绝，因为会和已存在 active change 产生任务重复，增加冲突和归档漂移。

### Decision 2: MMW GPS v2 保留 package CLI，退役旁支脚本

MMW GPS v2 主线保留：

- `kd-sensing-mmw-town-gps-v2`
- `kd-sensing-plot-mmw-town-gps-v2`
- `kd-sensing-compare-mmw-town-gps-v2`
- `configs/mmw_town_gps_adapter_v2.yaml`
- `src/kd_sensing/engine/mmw_town_gps_v2.py`
- `src/kd_sensing/cli/{mmw_town_gps_v2,plot_mmw_town_gps_v2,compare_mmw_town_gps_v2}.py`

退役对象是脚本层旁支：

- `scripts/mmw/visualize_gps_angle_beam_correspondence.py`
- `scripts/mmw/visualize_gps_prediction_trajectory.py`
- `scripts/mmw/visualize_prediction_error_label_distribution.py`

实现时先比较三个脚本和 package plotter 输出的图表/summary。如果某个图表仍是 current spec 要求，就把最小实现迁入 `kd_sensing.cli.plot_mmw_town_gps_v2` 或其 owner helper；若只是历史探索图，则删除脚本并在 inventory 记录为 retired historical diagnostic。

Alternative: 直接退役整个 MMW GPS v2。拒绝，因为它仍是 current `formal diagnostic`，有 spec、config、CLI 和 focused tests。

### Decision 3: AMR-Net_gps_image 退役为 tombstone，不再保留 mock runner

AMR-Net 当前价值主要是记录 IEEE `11282996` metadata 与 Scenario 23 作者包冲突。退役后保留：

- 文档中的 conflict 摘要、blocked official 边界和“不启用 LiDAR/不声明 official reproduction”的警示。
- OpenSpec tombstone requirement，说明旧 CLI/config/module path 不再作为 current 能力。
- Config/migration/architecture tests，确保 `configs/baselines/amr_net_gps_image.yaml`、`kd-sensing-run-amr-net-gps-image` 和 `kd_sensing.baselines.amr_net_gps_image` 不回流为 current。

删除或收口：

- `pyproject.toml` entry point。
- `src/kd_sensing/cli/run_amr_net_gps_image.py`。
- `configs/baselines/amr_net_gps_image.yaml`。
- `src/kd_sensing/baselines/amr_net_gps_image/` 运行实现。
- `tests/test_amr_net_gps_image.py` 中 current runner/mock metric 断言，改为 retired guard tests。
- `docs/mainline_model_catalog.md` 和 `docs/result_claims_registry.md` 的 current claim 行。

Alternative: 保留 source-audit-only CLI。拒绝，除非用户明确表示还要长期维护 AMR 论文审计；否则 source audit 也会继续占据 package CLI 和 claim registry 表面积。

### Decision 4: JEPA-MSAC 退役 current workflow，保留历史论文复现说明

JEPA-MSAC 当前没有 paper-aligned 长训练结果，smoke 只验证 schema。退役后保留：

- OpenSpec tombstone，说明 JEPA-MSAC 不再作为 current paper/workflow reproduction。
- 文档中的历史背景和“如需复核请查 git history/archive change”的提示。
- `jepa_msac` 相关 runtime output 只作为 cleanup manifest 候选，不自动删除。

删除或收口：

- `pyproject.toml` entry point `kd-sensing-run-jepa-msac`。
- `src/kd_sensing/cli/run_jepa_msac.py`。
- `configs/pretraining/jepa_msac_s32_smoke.yaml` 与 `configs/pretraining/jepa_msac_s32_paper.yaml`。
- `src/kd_sensing/baselines/jepa_msac/`。
- `src/kd_sensing/models/jepa_msac.py` 的 whole-model registry 例外。
- `src/kd_sensing/losses/jepa_msac.py` 和 `engine.objectives` 中只服务 JEPA-MSAC 的 objective/history/metadata key。
- `tests/test_jepa_msac.py` 的 current smoke tests，改为 retired config/CLI/module guard。
- README、experiment matrix、mainline catalog、protocols 和 result claim registry 中的 current 行。

实现时必须先用引用扫描确认 `jepa_msac_pretraining`、`val_jepa_msac_loss`、`JepaMsacModel` 和 `jepa_msac_stage2_losses` 没有被其它 current workflow 消费。若发现共享 helper，应先拆出通用部分再删除 JEPA-MSAC 专属命名。

Alternative: 降级为 diagnostic-only smoke。拒绝，因为它仍需要 model/loss/objective/config/CLI 一整套表面积；如果没有真实 paper-aligned 复现计划，维护成本高于价值。

### Decision 5: shell orchestration 只保留 CSI hardening，退役本地历史 wrapper

退役对象：

- `scripts/run_deepsense_gps_circular_soft_label.sh`
- `scripts/run_mmw_gps_circular_soft_label_ablation.sh`
- `scripts/run_mmw_sunny_modal15_l5p3_h123.sh`
- `scripts/run_mmw_sunny_modal15_l5p6_h246.sh`

保留对象：

- `scripts/run_csi_hardening_matrix.sh`

退役方式：

- 从 maintainer index `shell_allowlist` 删除。
- 从 project inventory 脚本 allowlist 删除或移入 historical note。
- 从 README/experiment matrix 删除推荐命令。
- 在 architecture boundary tests 中加入路径不存在/不在 allowlist 的断言。
- 不新增替代 shell wrapper；需要复跑实验时使用 `conda run -n kd_mm_beam kd-sensing-train ...`、当前 package CLI 或保留的 CSI hardening runner。

Alternative: 全部 shell runner 删除。拒绝，因为 CSI hardening matrix 是 current formal/debug matrix，shell runner仍承担矩阵编排价值。

## Risks / Trade-offs

- [Risk] 删除 AMR-Net/JEPA-MSAC 会丢失 blocked 官方复现背景。→ Mitigation: 将背景保留在 tombstone spec、result registry 历史说明或 project inventory，不保留 current runner。
- [Risk] JEPA-MSAC objective key 可能和通用 objective metadata 交织。→ Mitigation: 先跑引用扫描，删除前拆出通用 helper，focused tests 覆盖 objective metadata 和 config validation。
- [Risk] MMW 旁支脚本里有 package plotter 尚未覆盖的图。→ Mitigation: 先盘点图表输出；只迁移仍被 current spec/README 需要的最小图，其余退役。
- [Risk] 旧本地 notebook 或 shell history 仍调用被删入口。→ Mitigation: README/文档给出迁移方向；config load/CLI help/architecture tests 提供清晰失败边界。
- [Risk] 和 `retire-legacy-model-registry-surface` 同时修改文档导致冲突。→ Mitigation: 先完成/合并 registry change；本 change 后续文档以 registry change 归档后的 current 状态为准。

## Migration Plan

1. 前置检查：确认 `retire-legacy-model-registry-surface` apply/验证状态，必要时先完成该 change。
2. MMW 脚本收口：盘点三个 `scripts/mmw/visualize_gps_*` 输出；迁移必要图表到 package plotter；删除脚本；更新 script allowlist 和 docs。
3. AMR-Net 退役：删除 CLI/config/baseline package current 实现；把 tests 改为 retired guard；更新 docs/claims/spec tombstone。
4. JEPA-MSAC 退役：引用扫描专属模型/loss/objective/config；删除专属入口和实现；保留历史说明；更新 tests 和 docs。
5. Shell wrapper 退役：删除四个非 CSI shell wrappers；更新 maintainer index、project inventory、README/experiment matrix 和 architecture tests。
6. 文档账本收口：同步 mainline catalog、protocols、claim registry、experiment matrix、README 和 lifecycle inventory。
7. 验证：
   - `openspec validate retire-priority-legacy-surfaces --strict`
   - `openspec validate retire-legacy-model-registry-surface --strict`（若仍未归档）
   - `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
   - `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`
   - MMW 触碰时追加 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py -q`
   - JEPA/objective 触碰时追加 `conda run -n kd_mm_beam pytest tests/test_objective_metadata.py tests/test_prediction_objectives.py -q`

Rollback:

- 如果 AMR-Net 或 JEPA-MSAC 仍被用户指定为近期复现目标，回滚对应删除并把该入口降级为 `diagnostic-only` 或 `blocked-source-audit`，同时保留 CLI/config/test。
- 如果 MMW package plotter 无法覆盖必要图表，保留对应脚本为临时 `research_diagnostic`，但必须给出删除条件和 focused validation。
- 如果 shell wrapper 删除影响当前 CSI 以外的正式矩阵，恢复该 wrapper 并在 maintainer index 标注 owner、output boundary 和退役条件。

## Open Questions

- AMR-Net 的 metadata conflict 是否还需要独立 Markdown 记录，还是只保留在 OpenSpec tombstone 与 result claim registry 历史说明中？
- JEPA-MSAC 删除 `model.primary.type: jepa_msac` 后，是否需要 `MODELS.register_removed("jepa_msac", "...")` 给本地旧 config 更友好的错误？
- MMW 旁支脚本中的 angle-beam correspondence 图是否应成为 `kd-sensing-plot-mmw-town-gps-v2` 的可选图，还是作为历史 exploratory plot 退役？
