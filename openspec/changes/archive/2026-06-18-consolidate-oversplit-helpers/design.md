## Context

当前工作树显示上一轮 hotspot remediation 已经把多个 owner 模块拆成了大量内部 helper 文件。拆分本意是降低巨型模块风险，但现在局部出现相反问题：许多 helper 只有几十到一两百行，且只被同一个 owner 模块导入，阅读 JEPA benchmark 或 data factory 时需要在多个文件之间频繁跳转。

本 change 面向个人论文实验代码的维护体验，目标是减少低价值文件边界和重复防御式检查。它不能改变已有训练、评估、manifest、metrics、split 或输出 schema，也不能绕过 `src/kd_sensing` 包结构、恢复旧入口或新增兼容聚合层。

现有约束：

- `project-architecture` 要求职责清晰、轻量导入边界稳定、旧入口不回流。
- `jepa-gps-shortcut-benchmark` 要求 manifest validation、扰动 determinism、comparability metadata、输出表格和 benchmark manifest 可测试。
- 当前 active change `add-predictive-gps-query-advantage` 已完成 artifacts，未来可能继续扩展 JEPA benchmark 行为；本 change 只改布局与冗余检查，不抢 predictive 语义。
- 工作树已有未提交源码和文档改动；实现时必须只在目标文件上增量整理，不回滚用户已有改动。

## Goals / Non-Goals

**Goals:**

- 将同 owner、低复用、单调用点或只服务 re-export 的 helper 小文件合并回清晰 owner 模块。
- 删除内部重复断言、重复类型检查、重复 `None` 检查和只包一层再抛出的异常包装。
- 保持 public facade、CLI、console script、manifest schema、metrics schema、输出目录和真实实验产物边界不变。
- 更新治理索引、inventory 和架构边界测试，使新布局被测试认可。
- 通过 focused tests 验证行为没有变化。

**Non-Goals:**

- 不做全仓库“越少文件越好”的机械合并。
- 不把 `jepa_gps_shortcut_benchmark.py` 重新做成巨型 public facade。
- 不删除用户输入、manifest/config、文件路径、split/comparability、no-future-leak 和产物边界检查。
- 不改变 JEPA benchmark 的 P-suite/CxD/predictive robustness 语义。
- 不引入新依赖，不迁移 checkpoint，不生成或提交真实训练输出。

## Decisions

### Decision 1: 以 owner 模块为合并单位，而不是按目录横向聚合

优先合并同一 owner 内的 helper：

- `jepa_benchmark_common_types.py`、`jepa_benchmark_io.py`、`jepa_benchmark_metadata.py`、`jepa_benchmark_scalars.py` 合并回 `jepa_benchmark_common.py`。
- `jepa_benchmark_cxd_helpers.py`、`jepa_benchmark_cxd_phase.py`、`jepa_benchmark_cxd_dominance.py`、`jepa_benchmark_cxd_failure_modes.py`、`jepa_benchmark_scenario_d_metrics.py`、`jepa_benchmark_scenario_d_normalization.py` 合并到 `jepa_benchmark_scenario_d.py` 或最多拆成一个清晰的 Scenario D owner + 一个 plots/artifacts 依赖边界。
- `jepa_benchmark_runner_summary.py`、`jepa_benchmark_runner_sources.py`、`jepa_benchmark_runner_manifest.py` 合并回 `jepa_benchmark_runner.py`。
- `data_factory_validation.py` 可优先合并回 `data_factory.py`；`data_factory_scalers.py` 若只服务 build flow，可合并，否则保留为 scaler owner；`data_factory_groups.py` 和 `data_factory_protocols.py` 只有在合并后仍保持 split protocol 可读时才合并。
- `sequence_columns.py`、`sequence_metadata.py`、`sequence_splits.py`、`sequence_windows.py` 只合并低复用和单调用点 helper；窗口生成与 split 语义若合并后过长，可以保留。
- `image_ae_gps_*` 只回收薄 wrapper 和 reports/evaluation 小 helper；训练、dataset、model 若合并后变成难读大文件则保留。

备选方案是新增一个 `helpers.py` 或 `_utils.py` 聚合所有小函数。该方案减少文件数但弱化 owner 边界，容易变成新的兼容聚合层，因此不采用。

### Decision 2: public facade 只保留 re-export 和 CLI 兼容

`jepa_gps_shortcut_benchmark.py` 继续作为 public import/CLI compatibility facade，不吸收实现。合并发生在内部 owner 模块，例如 Scenario D owner 或 runner owner。这样能减少 helper 文件数，同时不让 facade 回到超大实现文件。

备选方案是把所有 benchmark 逻辑合回 public facade。该方案文件数最少，但会违反现有 facade budget 和公开入口边界，不采用。

### Decision 3: 防御式检查按边界分层处理

保留检查：

- manifest/config/CLI 用户输入的必需字段、类型、路径存在性和协议枚举。
- split、sample_count、label_space、metric_profile、difficulty digest 和 checkpoint provenance comparability。
- Scenario C/D no-future-leak、mask/replay metadata、扰动 determinism。
- 本地产物边界、输出路径计划和 ignored outputs 规则。

删除或收敛检查：

- 同一函数调用链中已经验证过的重复 `isinstance` 和 `None` 分支。
- 内部私有 helper 的重复 `assert`，尤其是测试 fixture 或 owner caller 已保证输入时。
- `try/except Exception as exc` 只为了把异常包成同义 `RuntimeError/ValueError` 的层。
- “空列表返回空结果”和上层已经处理的重复空值保护，除非该空值会改变输出 schema。

备选方案是完全按用户偏好删除大部分错误处理。该方案代码最短，但会破坏 manifest/spec 中多个 MUST reject 或 MUST record warning 的契约，因此只删除内部冗余检查。

### Decision 4: 治理预算随合并更新，接受中等 owner 模块

合并后某些 owner 模块会变长。治理索引应从“强制拆细”改为“accepted-size 或 merge-required 已完成”，并记录新的 line budget、rationale 和 focused validation commands。预算不是追求最小行数，而是防止 public facade 和跨领域 owner 失控。

备选方案是保留旧 line budget 并让测试继续要求拆出的 helper 文件存在。该方案会阻止本 change 达成目标，不采用。

## Risks / Trade-offs

- [Risk] 合并后 Scenario D 或 runner owner 文件偏长，后续修改冲突变多。  
  → Mitigation: 只合并同 owner 强相关 helper，并在文件内按段落组织；保留 plots、manifest validation 等仍有清晰外部边界的模块。

- [Risk] 删除检查导致错误信息不如以前友好。  
  → Mitigation: 只删内部重复检查；用户输入、manifest、路径、comparability、no-future-leak 和输出边界检查必须保留。

- [Risk] 与 `add-predictive-gps-query-advantage` 后续实现冲突。  
  → Mitigation: 本 change 不改 predictive 行为；实现时先完成纯 import/layout 调整，再让 predictive change 基于合并后的 owner 模块继续扩展。

- [Risk] 架构边界测试仍引用被删除 helper 文件。  
  → Mitigation: 同步更新 `docs/maintainer_context_index.yaml`、inventory 和 `tests/test_architecture_boundaries.py` 的 helper expectations。

## Migration Plan

1. 用 `rg` 和现有测试定位只被同 owner 使用的 helper 文件，建立删除/保留清单。
2. 先合并 JEPA benchmark common、Scenario D/CxD 和 runner helper，更新 imports 与 `__all__`。
3. 跑 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`。
4. 合并 data factory、sequence preprocessing 和 BeamBench 中低复用 helper，按触碰范围跑 focused tests。
5. 删除冗余内部检查，保留边界检查，并跑相关 tests。
6. 更新治理索引、inventory 和架构边界测试。
7. 跑 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 与 `openspec validate consolidate-oversplit-helpers --strict`。

Rollback 策略：若某个合并 owner 变得难以维护，恢复该 owner 的 helper 文件和 imports；不要回滚无关源码或用户已有改动。

## Open Questions

- Scenario D/CxD 是全部合并到 `jepa_benchmark_scenario_d.py`，还是保留一个 `jepa_benchmark_cxd_analysis.py` 作为中等 owner，需要在实现时根据最终行数和可读性决定。
- `data_factory_scalers.py` 是否仍应作为 normalization/scaler owner 保留，取决于它是否被 data factory 之外复用。
- BeamBench Image AE+GPS 的训练、dataset、model 文件是否合并，取决于 focused tests 和最终文件可读性；第一阶段可以只删除薄 wrapper。
