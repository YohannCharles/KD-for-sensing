## Context

JEPA visual analysis 和 JEPA GPS shortcut benchmark 是当前论文图、robustness matrix、GPS shortcut evidence 和 Predictive Robustness claim gate 的核心诊断入口。现有实现已经有部分窄模块，但 `jepa_visual_analysis.py` 和 `jepa_benchmark_runner.py` 仍承担过多职责，新增 suite 或 artifact 时容易把 schema、绘图、真实 forward 和报告逻辑耦合在一起。

## Goals / Non-Goals

**Goals:**
- 按职责拆分 visual analysis 与 benchmark runner 的内部模块。
- 保持 CLI、公开 facade、manifest schema、CSV/JSON/report 输出兼容。
- 防止 suite-specific helper 回流到公开 facade。
- 为 Predictive、Scenario C/D/CxD 和 real-forward 诊断留下 focused tests。

**Non-Goals:**
- 不改变任何 robustness 指标、DBA/top-k、claim gate 或 comparability 规则。
- 不新增真实数据依赖，不要求存在本地 checkpoint。
- 不替换 matplotlib/CSV/JSON 输出技术栈。

## Decisions

1. **公开入口保持稳定，内部 owner 细分。**
   CLI 和 facade 继续提供当前 public surface；实现迁入 `jepa_visual_*`、`jepa_benchmark_*` 或等价窄模块。

2. **artifact planning 与 writing 分离。**
   benchmark runner 先构建 artifact plan，再由 writer 写 CSV/JSON/manifest；visual analysis 也将 report/manifest builder 从模型分析 loop 中分离。

3. **suite dispatch 显式化。**
   Predictive、GPS-query advantage、Scenario C/D/CxD、legacy P0-P5 和 real-forward 分支必须通过显式 dispatch/helper 承载，避免在主 runner 中追加条件块。

4. **schema 兼容优先于减少行数。**
   拆分目标是降低耦合，不以文件数量或行数作为唯一 KPI；若字段兼容风险高，先登记预算并补测试。

## Risks / Trade-offs

- 输出字段遗漏 -> 用 focused tests 比较 `metrics_by_condition.csv`、robustness summary、predictive bundle 和 analysis manifest 的关键字段。
- facade 回流 -> 架构边界测试继续拒绝内部模块从 `jepa_gps_shortcut_benchmark.py` 导入 private helper。
- 拆分后 import 变重 -> 保持配置/manifest 读取路径不 eager import torch dataset 或模型权重。

## Migration Plan

1. 为现有 visual/benchmark outputs 捕获 focused fixture。
2. 先抽纯 writer/plan/helper，再移动 suite-specific 计算。
3. 保留原公开函数签名和返回 dict。
4. 更新 inventory 热点预算与 owner 列表。
5. 运行 `openspec validate modularize-jepa-diagnostics-suites --strict`、`conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py tests/test_jepa_gps_shortcut_benchmark.py tests/test_modality_difficulty.py tests/test_architecture_boundaries.py -q`。

## Open Questions

- `jepa_visual_analysis.py` 的 GPS-query evidence package 是否单独成 owner，还是继续使用现有 `gps_query_evidence.py`？
- real-forward diagnostics 是否需要独立 manifest/result dataclass，还是先保持 dict schema？
