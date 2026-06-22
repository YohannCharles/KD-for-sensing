# gps-query-effectiveness-visualization Specification

## Purpose
TBD - created by archiving change add-gps-query-attention-evidence-visuals. Update Purpose after archive.
## Requirements
### Requirement: GPS-query 有效性证据包输入
系统 MUST 提供离线 GPS-query 有效性证据包输入契约，用于声明模型、paired baseline、指标表、可选 benchmark manifest、可选逐样本 forward cache、输出目录和 claim gate 配置。输入契约 MUST 记录 split、scene set、seed、checkpoint selection、label space、metric profile、difficulty/condition 和模型 config/weights provenance。

#### Scenario: 读取 evidence config
- **WHEN** 用户运行 GPS-query 有效性可视化分析并传入 evidence config
- **THEN** 系统 MUST 解析 GPS-query 模型、paired baseline、strong anchor baseline、指标表路径和输出目录
- **AND** 系统 MUST 写出解析后的 `evidence_manifest.json`
- **AND** 系统 MUST 不修改输入 config、checkpoint、训练日志、split CSV 或 benchmark 输出

#### Scenario: 可比性字段缺失
- **WHEN** GPS-query 模型和 paired baseline 缺少 split、seed、label space、metric profile 或 checkpoint provenance
- **THEN** 系统 MUST 将该 pair 标记为不可进入 strict paired claim
- **AND** 系统 MUST 在 `evidence_manifest.json` 和 `report.md` 中记录缺失字段

### Requirement: Paired ablation 有效性指标
系统 MUST 以 paired ablation 作为 GPS-query 有效性的主证据。系统 MUST 在同 split、同 seed、同 checkpoint selection、同 metric profile 和同 condition 下比较 GPS-query 与 mean pooling 或同视觉 encoder baseline，并输出 clean/P0、P1-P5、Scene31、S32-S34 和总体指标 delta。

#### Scenario: 导出 paired delta 表
- **WHEN** 输入包含 GPS-query 与 paired baseline 的 P0-P5 wide 或 long 指标表
- **THEN** 系统 MUST 写出 `tables/paired_delta_by_condition.csv`
- **AND** 每行 MUST 包含 model_pair、condition、scene_group、metric、query_value、baseline_value、absolute_delta、relative_delta 和 comparability_status

#### Scenario: 导出 P0-P5 有效性热图
- **WHEN** paired delta 表至少包含一个可比 model pair 和两个以上 condition
- **THEN** 系统 MUST 导出 P0-P5 delta heatmap
- **AND** 图表 MUST 标注 pair 名称、scene group、metric、sample count 或来源表路径

#### Scenario: 非 paired anchor 只作参考
- **WHEN** strong non-JEPA 或 image+GPS baseline 参与比较
- **THEN** 系统 MUST 将其标记为 anchor baseline
- **AND** 系统 MUST 不把 anchor comparison 替代同结构 paired ablation claim

### Requirement: GPS-query attention 热点图
系统 MUST 为提供 attention diagnostics 的 GPS-query 模型导出注意力热点图。系统 MUST 支持 `[sample,time,query,patch]`、`[sample,query,patch]` 或等价 attention map，MUST 使用 token grid metadata 将 patch attention reshape 为二维 grid，并 MUST 支持 query/time 平均或指定 query/time 的展示方式。

#### Scenario: 导出 patch-grid heatmap
- **WHEN** GPS-query 模型提供可用 attention map 和 token grid metadata
- **THEN** 系统 MUST 写出 patch-grid heatmap 图
- **AND** 图中 MUST 标注 model、sample id、scene、condition、target beam、Top-k prediction、history frame 和 query aggregation

#### Scenario: 导出 image overlay 热点图
- **WHEN** attention map 可用且输入图像 tensor 或 image path 可恢复
- **THEN** 系统 MUST 将 attention grid resize 到图像尺寸并导出 image overlay
- **AND** overlay MUST 保留原图可辨识内容、attention 色条或透明度说明、sample id 和模型预测摘要

#### Scenario: attention 统计表
- **WHEN** attention map 可用
- **THEN** 系统 MUST 写出 `tables/attention_summary.csv`
- **AND** 表中 MUST 包含 attention entropy、effective patch count、query diversity、center-of-mass、time steps、query count、patch count 和 aggregation method

#### Scenario: attention 不可用降级
- **WHEN** 模型不提供 attention diagnostics、token grid metadata 缺失或 attention shape 不可解析
- **THEN** 系统 MUST 跳过该模型的 attention 图
- **AND** 系统 MUST 继续生成 paired metric 表和 report
- **AND** 系统 MUST 在 manifest 和 report 中记录 skipped reason

### Requirement: Query gain/regression case study
系统 MUST 基于逐样本 comparison table deterministic 选择 case study。Case group MUST 至少覆盖 `query_gain`、`query_regression`、`shared_near_miss` 和 `shared_failure`，并 MUST 为每个 case 输出机器可读 payload 和可视化面板。

#### Scenario: 选择 query gain 和 regression
- **WHEN** comparison table 包含 paired baseline 和 GPS-query 的逐样本 target、Top-k、DBA contribution 或等价 error metric
- **THEN** 系统 MUST 按固定 seed 和排序规则选择 `query_gain` 与 `query_regression` 样本
- **AND** 系统 MUST 写出 `tables/case_selection.csv`
- **AND** 选择表 MUST 包含 group、sample id、selection reason、baseline error、query error 和 metric delta

#### Scenario: 导出 case panel
- **WHEN** 选中样本具备 image、prediction 和可选 attention 输入
- **THEN** 系统 MUST 导出 case panel
- **AND** panel MUST 包含原图或图像序列、GPS-query attention overlay 或 skipped marker、paired model Top-k/probability 摘要、target beam 和 error/delta

#### Scenario: 包含失败案例
- **WHEN** comparison table 中存在所有模型远错或 GPS-query 明显退化样本
- **THEN** 系统 MUST 输出 `shared_failure` 或 `query_regression` case
- **AND** report MUST 将其标记为失败模式而不是成功证据

### Requirement: Claim gate 报告
系统 MUST 生成 claim gate，用于区分可报告结论、解释性证据和 caveat。Claim gate MUST 基于 paired delta、clean regression、P0-P5 robustness、case 覆盖、attention 可用性和 strict comparability 状态生成，不得只依据单张 attention 图判定有效。

#### Scenario: 生成 claim gate summary
- **WHEN** evidence package 完成 paired metric 和可用诊断聚合
- **THEN** 系统 MUST 写出 `tables/claim_gate_summary.csv` 或等价 JSON
- **AND** 每个 claim MUST 标记为 `supported`、`exploratory`、`insufficient` 或 `blocked`
- **AND** 每个 claim MUST 引用支撑表格或图表路径

#### Scenario: report 避免过度声称
- **WHEN** 系统写出 `report.md`
- **THEN** report MUST 分别列出 `reportable`、`interpretive` 和 `caveat` 结论
- **AND** report MUST 明确说明 attention hotspot 是解释性证据而非因果证明

### Requirement: 产物边界和可测试性
GPS-query 有效性证据包 MUST 将所有生成图表、表格、case payload、cache 和报告写入 ignored 的 `outputs/` 或用户显式指定的本地产物目录。系统 MUST 为核心计算、降级行为和 manifest schema 提供自动化测试，测试 MUST 使用 synthetic/mock 数据，不得读取真实 `dataset/`。

#### Scenario: 输出目录结构
- **WHEN** evidence package 完成
- **THEN** 输出目录 MUST 包含 `evidence_manifest.json`、`report.md`、`tables/`、`figures/` 和可选 `cases/`
- **AND** manifest MUST 记录命令、输入路径或 digest、模型 provenance、condition、seed、输出文件清单和 warnings

#### Scenario: synthetic 测试覆盖
- **WHEN** 单元测试使用 synthetic metrics、attention map 和 sample comparison rows
- **THEN** paired delta、attention reshape、case selection、claim gate 和 unavailable fallback MUST 产生可验证输出
- **AND** 测试 MUST 不依赖真实 checkpoint、真实 dataset 或 ignored runtime artifacts

