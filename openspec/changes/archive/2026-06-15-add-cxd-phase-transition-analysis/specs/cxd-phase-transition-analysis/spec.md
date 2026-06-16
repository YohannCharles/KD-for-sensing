## ADDED Requirements

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

### Requirement: CNN 与 JEPA crossing detection
系统 MUST 检测 CNN/AE/ResNet 类 image+GPS baseline 与 Image-JEPA/Image-JEPA+GPS/query-pool 模型之间的 crossing region。检测 MUST 只在 strict comparable 的 rows 上执行，且 MUST 输出 crossing condition、metric margin、参与配对模型、regime label 和 query_pool 相对 biased 的 shift summary。

#### Scenario: 检测 JEPA 超过 CNN 的 crossing region
- **WHEN** 同一 `(Cx, Dy, seed, split, metric profile)` 下存在可比较的 CNN+GPS 与 JEPA 模型指标
- **THEN** analysis MUST 标记 JEPA primary metric 大于 CNN primary metric 的 condition 为 crossing region
- **AND** analysis MUST 写出 `results/crossing_region_Cx_Dy.json`
- **AND** crossing summary MUST 记录 CNN best、JEPA best、metric margin 和 condition id

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
- **AND** 图表可用时 MUST 写出 `plots/cxd_accuracy_heatmap.png`、`plots/cnn_jepa_crossover_curve.png` 和 `plots/modality_dominance_heatmap.png`

#### Scenario: 图表依赖不可用时保留表格
- **WHEN** matplotlib 或等价可视化依赖不可用
- **THEN** system MUST 仍写出 CSV、JSON 和 NPY artifact
- **AND** runner manifest MUST 标记图表为 skipped
- **AND** CLI MUST 成功完成，除非必需的 metrics 输入不可用
