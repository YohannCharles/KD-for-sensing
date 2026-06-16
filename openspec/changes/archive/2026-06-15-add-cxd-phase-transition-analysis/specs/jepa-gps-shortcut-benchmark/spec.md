## ADDED Requirements

### Requirement: CxD analysis manifest schema
JEPA GPS shortcut benchmark manifest MUST 支持声明 CxD phase transition analysis 配置。配置 MUST 能声明 enabled flag、primary metric、paired model groups、dominance diagnostic sources、diagnostic fallback policy、failure-mode thresholds 和 output artifact plan。

#### Scenario: 解析 CxD analysis 配置
- **WHEN** benchmark manifest 包含 `analysis.cxd_phase_transition`
- **THEN** runner MUST 解析是否启用 phase diagram、dominance、crossing 和 failure decomposition
- **AND** runner MUST 校验 paired model references 是否存在于 `models`
- **AND** runner MUST 校验 diagnostic source path 或 model diagnostic field 的声明格式

#### Scenario: 拒绝不安全 fallback
- **WHEN** manifest 将 dominance fallback 设置为 heuristic-only formal evidence
- **THEN** runner MUST 拒绝该 manifest 或将 dominance status 降级为 smoke/unavailable
- **AND** 错误或 warning MUST 说明正式 dominance evidence 需要 gradient、attention、fusion weights、latent variance 或等价真实诊断

### Requirement: CxD runner output integration
JEPA GPS shortcut benchmark runner MUST 在执行 Scenario CxD joint suite 后调用 CxD phase transition analysis，并将新增 artifact 注册到 benchmark manifest。Runner MUST 保留现有 metrics、robustness summary、shortcut reliance 和 Scenario D 输出兼容性。

#### Scenario: runner manifest 登记 CxD outputs
- **WHEN** CxD phase analysis 生成新增 artifact
- **THEN** `benchmark_manifest.json` MUST 在 output registry 或 `output_files` 中登记每个 artifact 的相对路径、kind、status 和 skipped reason
- **AND** result dict MUST 返回新增核心文件路径或在 manifest 中可定位
- **AND** 旧的 `metrics_by_condition.csv`、`robustness_summary.csv` 和 `shortcut_reliance_summary.csv` MUST 继续写出

#### Scenario: dry-run 不读取真实 checkpoint
- **WHEN** runner 使用 `--dry-run` 或 smoke manifest
- **THEN** CxD analysis MUST 使用 synthetic/mock metrics 或已声明 input tables
- **AND** runner MUST 不读取真实 `dataset/`、不启动训练、不修改 checkpoint
- **AND** 输出状态 MUST 标记为 dry_run、mock 或 smoke

### Requirement: Dominance diagnostics ingestion
JEPA GPS shortcut benchmark MUST 能从模型 forward diagnostics、external diagnostic artifacts 或 manifest inline summary 中读取 dominance 诊断。诊断 ingestion MUST 与 task performance metrics 分开记录。

#### Scenario: 读取 external dominance artifact
- **WHEN** manifest 为某个模型声明 dominance diagnostics CSV、JSON 或 NPZ
- **THEN** runner MUST 校验该 artifact 的 model、condition、seed 和 split 字段可与 CxD rows 对齐
- **AND** runner MUST 将 contribution scores 写入 `results/modality_dominance.csv`
- **AND** 不匹配 rows MUST 标记为 unavailable 并写入 warning

#### Scenario: 模型 forward diagnostics 只读消费
- **WHEN** evaluation source 返回 attention、gradient 或 latent diagnostic metadata
- **THEN** runner MUST 只读消费这些 metadata
- **AND** runner MUST 不改变训练 run 目录、checkpoint、split CSV 或输入 manifest
- **AND** diagnostics 缺失时 MUST 不影响 performance metrics 输出

### Requirement: CxD no label shift guard
JEPA GPS shortcut benchmark MUST 保持 Scenario C GPS perturbation 与 Scenario D image observability perturbation 不移动 target label、beam power、soft target、sample id 或 split metadata。CxD analysis MUST 在 manifest 或 tests 中记录该 guard。

#### Scenario: synthetic batch label guard
- **WHEN** focused test 对 synthetic batch 应用 CxD difficulty conditions
- **THEN** `target_beam`、beam power、soft target、sample id 和 split metadata MUST 与输入保持一致
- **AND** gps/image corruption metadata MUST 独立记录 C/D condition
- **AND** CxD analysis MUST 使用 condition metadata 而不是改写 label 来计算 phase diagram

### Requirement: CxD benchmark focused tests
JEPA GPS shortcut benchmark MUST 提供 focused tests 覆盖 CxD phase aggregation、dominance unavailable、external diagnostics ingestion、crossing detection、failure decomposition、artifact manifest 和 visualization skipped fallback。所有项目相关 Python 测试 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: focused tests 验证 CxD analysis schema
- **WHEN** 开发者运行 CxD benchmark focused tests
- **THEN** tests MUST 不读取真实 `dataset/`
- **AND** tests MUST 不写入 checkpoint、cache 或真实 metrics 到源码目录
- **AND** tests MUST 验证新增 CSV/JSON/NPY artifact schema 与 manifest registration
