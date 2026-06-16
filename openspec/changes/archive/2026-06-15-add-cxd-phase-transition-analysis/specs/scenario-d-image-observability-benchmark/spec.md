## ADDED Requirements

### Requirement: Scenario D phase transition artifact
Scenario D benchmark MUST 将 CxD matrix 的基础 metrics 扩展为 phase transition artifact。该 artifact MUST 与现有 `results/scenario_d_image_observability.csv` 和 `results/heatmap_cx_dy.npy` 保持兼容，同时新增 phase diagram、crossing、dominance 和 failure decomposition 输出。

#### Scenario: Scenario D benchmark 输出 phase analysis 文件
- **WHEN** Scenario D CxD benchmark 完成至少一个 strict comparable model group
- **THEN** output root MUST 继续写出 `results/scenario_d_image_observability.csv`
- **AND** output root MUST 继续写出 `results/heatmap_cx_dy.npy`
- **AND** output root MUST 写出 `results/cxd_phase_diagram.csv`
- **AND** output root MUST 写出 `results/crossing_region_Cx_Dy.json`
- **AND** output root MUST 写出 `results/failure_mode_decomposition.csv`
- **AND** runner manifest MUST 在 `output_files` 中登记这些文件

#### Scenario: Scenario D smoke 不升级为研究 claim
- **WHEN** Scenario D benchmark 使用 synthetic metrics、mock weights 或 smoke manifest
- **THEN** all phase analysis outputs MUST 标记为 `mock/smoke` 或等价状态
- **AND** claim registry MUST 不记录真实性能数值
- **AND** docs MUST 明确真实 checkpoint matrix 仍需本地运行且输出不进入源码

### Requirement: Scenario D dominance evidence status
Scenario D benchmark MUST 区分 performance metric 与 dominance diagnostic evidence。缺少真实 diagnostics 的模型 MUST 仍可进入 CxD performance heatmap，但 MUST 不被写成有 dominance 解释证据。

#### Scenario: dominance 诊断字段可追踪
- **WHEN** Scenario D CxD rows 被写出
- **THEN** 每个 dominance row MUST 记录 model、group、gps_condition、image_condition、seed、diagnostic_source、diagnostic_status 和 unavailable reason
- **AND** performance-only rows MUST 不自动产生正式 contribution score

#### Scenario: attention 或 gradient 不可用
- **WHEN** 某个模型缺少 attention、gradient 或 latent diagnostics
- **THEN** Scenario D benchmark MUST 将该模型的 dominance status 标记为 unavailable 或 skipped
- **AND** benchmark MUST 继续输出 accuracy heatmap、RSI、worst-case 和 crossing 可用部分
- **AND** report 或 manifest MUST 标记该 dominance evidence 不足以支撑模态主导结论

### Requirement: Scenario D crossing boundary 可比较性
Scenario D benchmark MUST 只在严格可比较的模型和 condition rows 上计算 CNN vs JEPA crossing boundary。比较 MUST 保持同一 split、label space、metric profile、sample_count、seed 和 difficulty digest。

#### Scenario: 不可比较模型不进入 crossing boundary
- **WHEN** CNN 与 JEPA 模型的 split、metric profile、sample_count 或 difficulty digest 不一致
- **THEN** benchmark MUST 不把它们写入同一 crossing boundary
- **AND** manifest MUST 记录不可比较字段
- **AND** 对应 crossing result MUST 标记为 unavailable 或 isolated group

#### Scenario: query_pool crossing shift summary
- **WHEN** Scenario D benchmark 同时包含 GPS-biased JEPA 和 GPS-query-pool JEPA
- **THEN** benchmark MUST 输出 query_pool 相对 biased JEPA 的 crossing shift summary
- **AND** summary MUST 使用同一 metric 和同一 CxD grid
- **AND** 缺少任一配对模型时 MUST 标记为 unavailable
