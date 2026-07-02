# cxd-phase-transition-analysis Specification

## Purpose
定义当前 JEPA GPS shortcut / Scenario D benchmark 的 CxD phase transition analysis 契约：从同一 run 的 condition-level metrics 生成 CxD phase diagram、modality dominance、ResNet-vs-JEPA crossing detection、failure decomposition 和本地产物边界，用于支撑鲁棒性诊断而不替代训练入口或真实 claim provenance。
## Requirements
### Requirement: CxD phase diagram 聚合
系统 MUST 提供 CxD phase transition analysis，用于从同一 benchmark run 的 condition-level metrics 聚合 `C0_sync` 到 `C4_severe_async` 与 `D0_full_image` 到 `D7_joint_worst_case` 的二维 phase diagram。聚合 MUST 保留 model、group、seed、split、sample_count、primary metric、Top-1、Top-3、DBA、clean metric、clean delta、relative drop、RSI、difficulty digest 和 comparability status。

#### Scenario: 生成完整 CxD phase diagram
- **WHEN** benchmark metrics 包含某个模型的完整 CxD joint suite
- **THEN** analysis MUST 输出该模型 5x8 个 CxD condition rows
- **AND** 每行 MUST 记录 `gps_condition`、`image_condition`、`c_severity`、`d_severity`、primary metric、Top-1、Top-3、DBA、clean delta、relative drop 和 RSI
- **AND** analysis MUST 写出 `results/cxd_phase_diagram.csv`
- **AND** analysis MUST 写出模型维度可追踪的 `results/cxd_phase_heatmap.npy`

#### Scenario: 缺少 CxD 条件时标记不可完整
- **WHEN** 某个模型缺少一个或多个 CxD condition
- **THEN** analysis MUST 保留已存在的 rows
- **AND** summary MUST 将该模型标记为 `incomplete_cxd_grid`
- **AND** crossing、surface integral 和 failure decomposition MUST 不把缺失条件补零或静默插值为真实指标

### Requirement: Modality dominance 诊断
系统 MUST 计算或记录每个模型和每个 CxD condition 的 modality dominance 诊断。诊断字段 MUST 至少包含 `gps_contribution_score`、`image_contribution_score`、`jepa_latent_contribution_score`、`diagnostic_source`、`diagnostic_status` 和 unavailable reason。正式 dominance 分析 MUST 来自 gradient norm、attention/fusion weights、JEPA latent variance 或 manifest 声明的等价真实诊断，不得仅用模型组启发式冒充解释性证据。

#### Scenario: 使用 gradient norm 计算 GPS 和 image contribution
- **WHEN** analysis 收到同一 batch/condition 下的 `gps_gradient_norm` 和 `image_gradient_norm`
- **THEN** `gps_contribution_score` MUST 计算为 `gps_gradient_norm / (gps_gradient_norm + image_gradient_norm)`
- **AND** `image_contribution_score` MUST 使用同一分母归一化
- **AND** 分母为零或缺失时该 condition MUST 标记为 `diagnostic_unavailable`

#### Scenario: 使用 attention 或 fusion weights 记录 dominance
- **WHEN** 模型 diagnostics 提供 GPS token、image token 或 JEPA latent token 的 attention/fusion weights
- **THEN** analysis MUST 将权重聚合到与 CxD condition 对齐的 dominance rows
- **AND** output MUST 记录 attention/fusion tensor 的来源、聚合口径和是否跨 head/query/time 平均
- **AND** attention/fusion weights MUST 被标记为解释性诊断，不得单独写成因果证明

#### Scenario: 诊断不可用时安全降级
- **WHEN** 某个模型不提供 gradient、attention、fusion weights 或 latent diagnostics
- **THEN** analysis MUST 继续输出 performance phase diagram
- **AND** `results/modality_dominance.csv` 中对应 rows MUST 标记 `diagnostic_status=unavailable`
- **AND** runner manifest MUST 记录 unavailable reason

### Requirement: ResNet 与 JEPA crossing detection
系统 MUST 检测 ResNet/AE/ResNet 类 image+GPS baseline 与 Image-JEPA/Image-JEPA+GPS/query-pool 模型之间的 crossing region。检测 MUST 只在 strict comparable 的 rows 上执行，且 MUST 输出 crossing condition、metric margin、参与配对模型、regime label 和 query_pool 相对 biased 的 shift summary。

#### Scenario: 检测 JEPA 超过 ResNet 的 crossing region
- **WHEN** 同一 `(Cx, Dy, seed, split, metric profile)` 下存在可比较的 Image ResNet+GPS 与 JEPA 模型指标
- **THEN** analysis MUST 标记 JEPA primary metric 大于 ResNet primary metric 的 condition 为 crossing region
- **AND** analysis MUST 写出 `results/crossing_region_Cx_Dy.json`
- **AND** crossing summary MUST 记录 ResNet best、JEPA best、metric margin 和 condition id

#### Scenario: 区分低退化和高退化 regime
- **WHEN** crossing summary 已生成
- **THEN** analysis MUST 将 clean 或低 severity 区域标记为 `low_degradation_regime`
- **AND** 将 JEPA 稳定占优的高 severity 区域标记为 `jepa_robust_regime`
- **AND** 无稳定 crossing 的模型对 MUST 标记为 `no_crossing_detected`

#### Scenario: 比较 query_pool 与 biased JEPA crossing shift
- **WHEN** manifest 或模型组中同时包含 GPS-biased JEPA 和 GPS-query-pool JEPA
- **THEN** analysis MUST 比较两者进入 JEPA robust regime 的最早 CxD condition
- **AND** output MUST 记录 `query_pool_shift` 为 earlier、same、later 或 unavailable

### Requirement: Failure mode decomposition
系统 MUST 基于 clean、GPS-only axis、image-only axis 和 joint CxD 指标拆解失败模式。分类 MUST 至少包含 `gps_fail_dominant`、`image_fail_dominant`、`both_fail`、`superadditive_joint_fail` 和 `unavailable`。

#### Scenario: 使用单轴参考拆解 CxD worst-case
- **WHEN** analysis 有 `(C0,D0)`、`(Cx,D0)`、`(C0,Dy)` 和 `(Cx,Dy)` 的主指标
- **THEN** analysis MUST 计算 GPS-only drop、image-only drop 和 joint drop
- **AND** analysis MUST 写出 `results/failure_mode_decomposition.csv`
- **AND** `C4_severe_async + D7_joint_worst_case` MUST 单独记录为 worst-case decomposition row

#### Scenario: 缺少单轴参考时标记 unavailable
- **WHEN** failure decomposition 缺少 clean、GPS-only 或 image-only reference row
- **THEN** analysis MUST 将该 condition 标记为 `unavailable`
- **AND** output MUST 记录缺失 reference 的 condition id
- **AND** 系统 MUST 不用其它模型或其它 split 的参考值替代

### Requirement: 论文图与产物边界
系统 MUST 将 CxD phase transition analysis 的真实运行产物写入 ignored 的 `outputs/analysis/...` 或 manifest 指定本地目录。源码变更 MUST 只包含实现、测试、配置、OpenSpec 和文档账本摘要，不得包含真实 CSV、NPY、PNG、checkpoint、cache 或 log。

#### Scenario: 写出机器可读 artifact 和图表
- **WHEN** CxD phase analysis 完成
- **THEN** output dir MUST 包含 `results/cxd_phase_diagram.csv`
- **AND** output dir MUST 包含 `results/cxd_phase_heatmap.npy`
- **AND** output dir MUST 包含 `results/modality_dominance.csv`
- **AND** output dir MUST 包含 `results/crossing_region_Cx_Dy.json`
- **AND** output dir MUST 包含 `results/failure_mode_decomposition.csv`
- **AND** 图表可用时 MUST 写出 `plots/cxd_accuracy_heatmap.png`、`plots/resnet_jepa_crossover_curve.png` 和 `plots/modality_dominance_heatmap.png`

#### Scenario: 图表依赖不可用时保留表格
- **WHEN** matplotlib 或等价可视化依赖不可用
- **THEN** system MUST 仍写出 CSV、JSON 和 NPY artifact
- **AND** runner manifest MUST 标记图表为 skipped
- **AND** CLI MUST 成功完成，除非必需的 metrics 输入不可用

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
