## Context

当前 JEPA GPS shortcut benchmark 已覆盖 GPS collapse、image degradation、Scenario C async GPS、Scenario D image observability、CxD matrix、Predictive Robustness、mock/smoke schema 和 visual-analysis ingestion。功能边界很丰富，但实现集中在单个 runner 文件中，导致新增 suite 或修 schema 时需要穿过大量无关 helper。

## Goals / Non-Goals

**Goals:**

- 将 runner 拆成按 suite 和 artifact 职责组织的窄模块。
- 保留原模块作为 facade，公开函数、CLI target 和 import path 不变。
- 为输出 schema 建立 regression checks，保证拆分前后 mock/smoke manifest 的 CSV/JSON 字段稳定。
- 降低 `jepa_gps_shortcut_benchmark.py` 文件预算，并防止 suite-specific helper 回流。

**Non-Goals:**

- 不新增新的 benchmark suite。
- 不改变真实评估、训练、metric 计算或 difficulty operator 语义。
- 不重命名公开 CLI、manifest key 或输出文件。
- 不移动本地 `outputs/analysis/` 产物。

## Decisions

### Decision 1: 保留 facade，拆出内部职责模块

原模块继续导出 `run_jepa_gps_shortcut_benchmark`、manifest validation、analysis bundle reader 和当前测试依赖的公开 helper。内部实现迁移到：

- `jepa_benchmark_manifest.py`: manifest/schema normalization 和 comparability checks。
- `jepa_benchmark_scenario_c.py`: async GPS/temporal drift suite。
- `jepa_benchmark_scenario_d.py`: image observability suite 和 CxD grid row generation。
- `jepa_benchmark_predictive.py`: Predictive JEPA P0-P5 normalization 和 metric rows。
- `jepa_benchmark_artifacts.py`: output registry、CSV/JSON writer、runner manifest。
- `jepa_benchmark_plots.py`: optional plotting 和 graceful fallback。

这样兼容外部 import，同时让内部改动更容易落在正确模块。

### Decision 2: 先移动纯函数，再移动 orchestrator

第一阶段迁移无状态纯 helper，例如 row aggregation、condition normalization、CSV scalar conversion、artifact path planning。第二阶段再移动 suite runner 和 output writer。这样每一步都能用现有 smoke tests 验证。

### Decision 3: 用 schema regression 防止行为漂移

拆分后必须比较 smoke manifest 的关键输出字段，包括 `benchmark_manifest.json`、`metrics_by_condition.csv`、`robustness_summary.csv`、Scenario D/CxD result 文件和 Predictive Robustness schema 字段。测试不比较真实数值 claim，只比较 mock/smoke schema 和 deterministic rows。

## Risks / Trade-offs

- [Risk] 移动 helper 时改变 CSV 列顺序或空值表示。  
  → Mitigation: 增加 focused regression，固定 mock/smoke 输出字段、必需列和 manifest keys。

- [Risk] facade 继续变厚。  
  → Mitigation: 架构边界测试限制 facade 行数，并拒绝 suite-specific helper 回流。

- [Risk] 拆分后循环 import。  
  → Mitigation: 让 manifest/artifact/schema helper 位于底层，suite 模块只依赖底层 helper，facade 只依赖 suite/orchestrator。

- [Risk] 一次性搬动太多难审查。  
  → Mitigation: tasks 按 suite 分阶段，每阶段运行 focused tests。
