## ADDED Requirements

### Requirement: JEPA benchmark runner 必须按 suite 和 artifact 边界拆分
JEPA GPS shortcut benchmark runner MUST 将 manifest/comparability loading、metric source ingestion、suite dispatch、aggregation、predictive artifact planning、output writing 和 runner manifest construction 拆分到窄 owner helper 或模块。

#### Scenario: benchmark 核心输出兼容
- **WHEN** 用户运行 `kd-sensing-jepa-gps-shortcut-benchmark`
- **THEN** `metrics_by_condition.csv`、`robustness_summary.csv`、`shortcut_reliance_summary.csv` 和 `benchmark_manifest.json` MUST 保持核心字段和注册行为

#### Scenario: suite-specific helper 不回流 facade
- **WHEN** 新增或修改 Scenario C/D/CxD/Predictive benchmark helper
- **THEN** 实现 MUST 位于对应窄 `jepa_benchmark_*` owner
- **AND** `jepa_gps_shortcut_benchmark.py` MUST remain a public facade rather than a suite implementation module
