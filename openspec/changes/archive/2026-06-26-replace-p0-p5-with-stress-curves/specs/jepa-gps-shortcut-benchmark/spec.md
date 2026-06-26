## ADDED Requirements

### Requirement: Predictive stress curve suite
Benchmark MUST support a predictive stress curve suite that evaluates clean anchor plus single-axis severity sweeps for image missing, image noise/degradation and GPS noise/unreliability. Each stress curve MUST isolate one perturbation axis and MUST reuse the shared difficulty pipeline.

#### Scenario: 默认 stress preset
- **WHEN** manifest 声明 predictive stress canonical preset
- **THEN** runner MUST expand it into clean anchor plus `image_missing`、`image_noise` and `gps_noise` suites
- **AND** each expanded suite MUST include severity values, severity unit, seed, split, operator params and difficulty digest

#### Scenario: image missing sweep
- **WHEN** runner evaluates `image_missing`
- **THEN** image tensor shape MUST remain unchanged
- **AND** missing frames MUST be expressed through zero-fill or configured sentinel plus `image_valid_mask=false` and `image_observability_score=0`
- **AND** GPS input, beam target, sample id and split metadata MUST remain unchanged

#### Scenario: image noise sweep
- **WHEN** runner evaluates `image_noise`
- **THEN** image input MUST be perturbed by one configured visual degradation axis only
- **AND** GPS input, beam target, sample id and split metadata MUST remain unchanged
- **AND** output metadata MUST record degradation type, severity, affected frame range and replay seed

#### Scenario: gps noise sweep
- **WHEN** runner evaluates `gps_noise`
- **THEN** GPS input MUST be perturbed by one configured GPS unreliability axis only
- **AND** image input, beam target, sample id and split metadata MUST remain unchanged
- **AND** output metadata MUST record GPS perturbation mode, severity, mask/delay/counterfactual fields and replay seed

#### Scenario: optional joint stress
- **WHEN** manifest explicitly enables `joint_stress`
- **THEN** runner MAY combine image missing and GPS noise at matched severity values
- **AND** output MUST label joint rows as diagnostic rather than primary claim rows

## MODIFIED Requirements

### Requirement: Benchmark 指标和论文图产物
Benchmark MUST 输出结构化指标和论文图产物。指标 MUST 至少包含 clean 指标、每个扰动条件下的 Top-K、DBA 或当前 objective 正式指标、相对下降、retention、collapse severity、area-under-robustness-curve 和可比较性 metadata。

#### Scenario: 写出鲁棒性汇总表
- **WHEN** benchmark 完成至少一个模型和一个扰动 suite
- **THEN** 输出目录 MUST 包含 `metrics_by_condition.csv` 或等价表格
- **AND** 输出目录 MUST 包含 `robustness_summary.csv` 或等价汇总
- **AND** 每行 MUST 记录 model、suite、condition、severity、seed、split、sample_count、primary metric、clean delta 和 retention

#### Scenario: 写出 stress 上限指标
- **WHEN** benchmark 完成 predictive stress curve suite
- **THEN** `robustness_summary.csv` 或等价汇总 MUST 包含 `S@drop<=0.02`、`S@drop<=0.05`、`AUC_retention`、`collapse_s` 和 `weakest_axis`
- **AND** 缺少 clean anchor 或 strict comparable rows 时，对应字段 MUST 标记为 unavailable 或 not-comparable

#### Scenario: 导出论文曲线
- **WHEN** benchmark 启用 figure export
- **THEN** 系统 MUST 导出 GPS noise/dropout 曲线、image missing 曲线、image degradation 曲线或 temporal delay 曲线中已配置的图表
- **AND** 图表 MUST 标注模型名、split、样本数、metric、severity 单位和 seed 或 digest
