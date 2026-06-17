## ADDED Requirements

### Requirement: Benchmark runner suite 模块化边界
JEPA GPS shortcut benchmark runner MUST 保持公共 CLI 和输出 schema 兼容，同时将 suite-specific normalization、metric row construction、aggregation 和 artifact planning 拆分到职责明确的窄 helper 或模块。新增 suite 不得继续无边界扩大单一 runner facade。

#### Scenario: predictive suite helper 可独立测试
- **WHEN** runner 支持 `predictive_jepa_robustness` suite
- **THEN** predictive condition normalization、predictive metric row construction 和 predictive regional aggregation MUST 位于可单独导入测试的 helper 或窄模块中
- **AND** existing `run_jepa_gps_shortcut_benchmark` facade MUST 继续返回兼容 result dict 和 output_files

#### Scenario: 拆分不改变输出 schema
- **WHEN** benchmark runner 内部 helper 被拆分
- **THEN** `metrics_by_condition.csv`、`robustness_summary.csv`、`shortcut_reliance_summary.csv`、predictive summary JSON/CSV 和 `benchmark_manifest.json` 的核心字段 MUST 保持兼容
- **AND** focused tests MUST 验证旧 manifest 和 predictive smoke manifest 的 output registration

### Requirement: Runner 热点预算和暂缓理由
若 implementation 阶段无法安全拆分 benchmark runner，项目 MUST 在 `docs/project_surface_inventory.md` 登记新的热点预算、拆分方向和暂缓原因。暂缓登记 MUST 不替代未来拆分，但 MUST 防止热点静默扩大。

#### Scenario: 拆分暂缓但 inventory 更新
- **WHEN** implementation 判断 `jepa_gps_shortcut_benchmark.py` 拆分风险超过本 change 范围
- **THEN** inventory MUST 记录当前规模、suite-specific 拆分方向、暂缓原因和后续优先级
- **AND** 架构边界测试 MUST 能防止该 runner 在未登记的情况下继续显著扩大

#### Scenario: 后续新增 suite 前先处理预算
- **WHEN** 后续 change 计划为 benchmark runner 新增 suite、analysis family 或 artifact family
- **THEN** 维护者 MUST 先确认 runner 已拆分到窄模块或 inventory 中有明确预算和拆分任务
- **AND** 新增 suite MUST 不复制已有 difficulty corruption、aggregation 或 writer 逻辑
