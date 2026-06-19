## ADDED Requirements

### Requirement: Benchmark 内部布局合并保持契约
JEPA GPS shortcut benchmark MAY 将同 owner 的内部 helper 合并为更少 Python 模块，但 public facade、manifest schema、suite normalization、perturbation semantics、comparability metadata、metrics CSV、benchmark manifest、图表产物和 runner CLI 行为 MUST 保持兼容。合并 MUST 不改变 P-suite、Scenario C、Scenario D、CxD 或 predictive robustness 的指标口径。

#### Scenario: Facade 行为保持不变
- **WHEN** 内部 `jepa_benchmark_*` helper 文件被合并或删除
- **THEN** `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` MUST 继续暴露当前 public benchmark 符号
- **AND** `kd-sensing-jepa-gps-shortcut-benchmark --help` MUST 继续可用
- **AND** public facade MUST 不吸收 benchmark runner、Scenario C/D、plotting 或 artifact registry 的主体实现

#### Scenario: Manifest 和输出 schema 保持不变
- **WHEN** benchmark runner 使用合并后的内部模块读取同一个 manifest
- **THEN** manifest validation、normalized suite config、metrics_by_condition、robustness_summary、benchmark_manifest 和 warnings 字段 MUST 与合并前语义兼容
- **AND** 未修改输入 manifest、训练配置、checkpoint、split CSV 或真实数据

#### Scenario: Scenario D 和 CxD 合并保持指标语义
- **WHEN** Scenario D/CxD normalization、phase diagram、dominance、failure-mode decomposition 或 metric-row helper 合并到更少 owner 模块
- **THEN** `scenario_d_image_observability` 与 `scenario_c_x_d_image_observability` suites MUST 继续输出相同条件字段、seed、difficulty digest、sample_count、metric、clean delta 和 comparability metadata
- **AND** 图表生成失败时仍 MUST 写出 metrics/manifest 并记录 warning

#### Scenario: Runner helper 合并保持产物边界
- **WHEN** runner summary、metric source ingestion 或 runner manifest helper 合并回 benchmark runner owner
- **THEN** evaluation-only、train-then-evaluate 和 reuse-existing-runs 协议 MUST 继续写入 ignored output directory 或 manifest 指定目录
- **AND** runner MUST 继续记录命令、环境、manifest digest、git status 摘要、模型配置、checkpoint provenance、difficulty provenance 和输出文件清单

### Requirement: Benchmark 冗余检查精简边界
JEPA GPS shortcut benchmark MAY 删除内部聚合、排序、标量转换和 row 派生 helper 中重复的二次检查，但 manifest validation、model comparability、suite normalization、perturbation determinism、Scenario C no-future-leak、Scenario D replay metadata 和 output artifact planning 的边界检查 MUST 保留。

#### Scenario: 内部 row helper 精简
- **WHEN** metric row、phase diagram、dominance ratio 或 summary helper 只消费 runner 已标准化的 rows
- **THEN** 实现 MAY 直接依赖标准化字段，删除重复类型检查和同义异常包装
- **AND** benchmark focused tests MUST 继续覆盖 summary rows、CxD rows 和 predictive rows 的核心字段

#### Scenario: 边界检查仍拒绝不可比较输入
- **WHEN** manifest 中模型的 split、sample_count、label_space、metric_profile、normalization artifact、difficulty profile digest 或 checkpoint provenance 不一致
- **THEN** benchmark MUST 继续拒绝写入同一严格可比较汇总或标记为不可比较
- **AND** 报告或 manifest MUST 记录不一致字段

#### Scenario: 扰动安全检查仍可测试
- **WHEN** Scenario C、Scenario D、CxD 或 predictive robustness suite 在 synthetic batch 上运行
- **THEN** repeated run with same seed MUST 保持 deterministic
- **AND** target label、sample id、未声明扰动的 modality 和 no-future-leak 约束 MUST 保持不变
