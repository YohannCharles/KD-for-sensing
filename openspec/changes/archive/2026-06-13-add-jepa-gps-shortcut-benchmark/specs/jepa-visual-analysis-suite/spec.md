## ADDED Requirements

### Requirement: Benchmark perturbation manifest 分析输入
JEPA visual analysis MUST 能读取 JEPA GPS shortcut benchmark manifest 或 benchmark runner 输出的机器可读 manifest，并将其中声明的模型、扰动 suite、severity、seed、split metadata 和指标产物纳入离线分析。分析入口 MUST 保持只读训练产物。

#### Scenario: 读取 benchmark manifest
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --analysis-config <path>` 且分析配置引用 benchmark manifest
- **THEN** 分析流程 MUST 读取 benchmark 的模型列表、扰动条件、severity sweep、metrics 表和 warnings
- **AND** 输出的 `analysis_manifest.json` MUST 记录 benchmark manifest 路径或 digest
- **AND** 分析流程 MUST 不修改 benchmark 输入表、训练 checkpoint、训练日志或 split CSV

#### Scenario: benchmark manifest 缺少可选图表输入
- **WHEN** benchmark manifest 未提供 attention、embedding 或 case payload 所需字段
- **THEN** 分析流程 MUST 跳过对应图表
- **AND** `analysis_manifest.json` 和 `report.md` MUST 记录 skipped reason
- **AND** 已具备输入的鲁棒性表和曲线 MUST 继续生成

### Requirement: Benchmark robustness matrix 图表和表格
JEPA visual analysis MUST 支持从 benchmark 指标表生成跨模型、跨扰动 suite、跨 severity 的 robustness matrix、collapse curve 和 clean-delta summary。输出 MUST 保持与现有 `figures/`、`tables/` 和 `report.md` 结构一致。

#### Scenario: 导出跨模型 robustness matrix
- **WHEN** benchmark 指标表包含两个或更多模型和一个或更多扰动 suite
- **THEN** 分析流程 MUST 写出跨模型 robustness matrix 表
- **AND** 表中 MUST 包含 clean metric、perturbed metric、delta、relative drop、sample_count、suite、condition 和 severity

#### Scenario: 导出 GPS collapse 曲线
- **WHEN** benchmark 指标表包含 GPS noise、GPS missing、GPS drift 或 GPS distractor suite
- **THEN** 分析流程 MUST 导出对应 collapse curve
- **AND** 图表 MUST 标注模型名、severity 单位、metric、split、样本数和 seed 或 digest

#### Scenario: 导出 image degradation 曲线
- **WHEN** benchmark 指标表包含 fog/rain、night、occlusion 或 motion blur suite
- **THEN** 分析流程 MUST 导出 image degradation robustness curve
- **AND** 图表 MUST 将 physical degradation type 与普通 augmentation 说明区分记录在 metadata 中

### Requirement: Shortcut reliance 报告段落
JEPA visual analysis MUST 在报告中单独总结 GPS shortcut reliance 相关发现，包括 drop GPS、misleading GPS、temporal delay、GPS-only collapse slope、JEPA 与 GPS-centric baseline 的 clean-delta 和 caveat。报告 MUST 明确区分性能结果、反事实 intervention 和解释性诊断。

#### Scenario: 报告 GPS shortcut 结论
- **WHEN** benchmark 产物包含 drop GPS 或 misleading GPS 条件
- **THEN** `report.md` MUST 包含 GPS shortcut reliance 小节
- **AND** 小节 MUST 引用对应表格或图表路径
- **AND** 小节 MUST 标记哪些结论来自 counterfactual intervention

#### Scenario: 报告避免过度声称
- **WHEN** attention、gradient 或 ablation reliance 诊断被用于解释模型行为
- **THEN** `report.md` MUST 将其标记为解释性证据
- **AND** 报告 MUST 不把 attention 或 gradient 单独描述为因果证明

### Requirement: Benchmark case study 选择
JEPA visual analysis MUST 支持根据 benchmark 条件选择 case study，至少覆盖 JEPA 在 GPS collapse 下优于 GPS-centric baseline 的 `jepa_recovery`、GPS-centric baseline 在 clean GPS 下占优但在 distractor 下失败的 `gps_shortcut_failure`、以及所有模型失败的 `shared_failure`。

#### Scenario: 选择 JEPA recovery case
- **WHEN** comparison table 中存在 JEPA 模型在 GPS collapse 条件下保持 Top-K hit 而 GPS-centric baseline 失败的样本
- **THEN** 分析流程 MUST 按 deterministic seed 和排序规则选择 case
- **AND** 系统 MUST 写出 case selection 表和机器可读 case payload

#### Scenario: 包含 shared failure
- **WHEN** benchmark 条件中存在所有模型均远错或指标显著下降的样本
- **THEN** 分析流程 MUST 可选择 `shared_failure` case
- **AND** report MUST 将其标记为失败模式，而不是只展示成功案例

### Requirement: Benchmark 分析降级与可测试性
JEPA visual analysis MUST 为 benchmark manifest ingestion、robustness matrix 生成、shortcut report 和缺失可选诊断的降级行为提供测试。缺少 optional visualization dependency 或样本数不足时，分析流程 MUST 记录 warning 并尽可能输出已有表格。

#### Scenario: 缺少可视化依赖
- **WHEN** matplotlib、UMAP、Grad-CAM 或其它 optional visualization dependency 不可用
- **THEN** 分析流程 MUST 写出可生成的 CSV/JSON 表格
- **AND** manifest 和 report MUST 记录图表跳过原因
- **AND** CLI MUST 成功完成，除非必需的 benchmark 指标输入不可用

#### Scenario: mock benchmark manifest 测试
- **WHEN** 单元测试使用 mock benchmark manifest 和小型 metrics 表运行分析 helper
- **THEN** 系统 MUST 生成 analysis manifest、robustness summary 表和 report skeleton
- **AND** 输出文件清单中的生成项 MUST 存在或被标记为 skipped
