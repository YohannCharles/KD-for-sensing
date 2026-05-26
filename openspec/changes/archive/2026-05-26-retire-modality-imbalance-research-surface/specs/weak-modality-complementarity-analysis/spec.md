## REMOVED Requirements

### Requirement: 互补分析命令入口
**Reason**: 弱模态互补样本分析依赖已退役的 Conditional Utility Audit，且研究线已放弃。
**Migration**: 不提供兼容入口；如需新的 case mining，应新建 capability。

#### Scenario: 互补分析命令入口退役
- **WHEN** 用户查找 `scripts/analysis/build_complementarity_cases.py`
- **THEN** 系统不再要求提供该脚本或 `complementarity_*` 输出

### Requirement: 输入 schema 适配
**Reason**: 互补分析输入 schema 只服务于已退役 case mining。
**Migration**: 删除相关 schema adapter；新的分析自行定义输入 schema。

#### Scenario: schema adapter 退役
- **WHEN** 用户提供历史 `subset_predictions`
- **THEN** 系统不再要求自动适配为互补分析内部 schema

### Requirement: 逐样本 case 表
**Reason**: 逐样本互补 case 表属于已放弃研究产物。
**Migration**: 不迁移历史 case 表；本地已有输出可作为静态文件保留。

#### Scenario: case 表退役
- **WHEN** 普通评估运行
- **THEN** 系统不再要求生成 `complementarity_cases.csv.gz`

### Requirement: case 判定语义
**Reason**: `rescue`、`unused_complementary` 和 `negative_transfer` 等标签属于已退役研究解释口径。
**Migration**: 不保留兼容判定 helper。

#### Scenario: case 标签退役
- **WHEN** 用户查看诊断 API
- **THEN** 系统不再要求提供互补 case 判定语义

### Requirement: summary 指标
**Reason**: 互补率、rescue rate 和 net fusion gain 属于已退役研究指标。
**Migration**: 使用普通评估 metrics 替代研究 summary。

#### Scenario: 互补 summary 退役
- **WHEN** case mining 不再提供
- **THEN** 系统不再要求生成 `complementarity_summary.json`

### Requirement: bucket 统计与报告
**Reason**: bucket 下互补统计和 markdown 报告依赖已退役 case 表。
**Migration**: 不迁移；普通 viewer diagnostics 保留。

#### Scenario: bucket 报告退役
- **WHEN** 用户查看互补分析输出目录
- **THEN** 系统不再要求生成 `complementarity_by_bucket.csv` 或 `complementarity_report.md`

### Requirement: Complementarity Explorer 页面
**Reason**: Gradio Complementarity Explorer 是已退役互补分析的前端。
**Migration**: 使用现有 manifest viewer 的样本浏览、raw/processed modalities、diagnostics 和 future distribution Tab。

#### Scenario: Explorer 页面退役
- **WHEN** 用户启动 Gradio viewer
- **THEN** 系统不再要求显示 Complementarity Explorer Tab

### Requirement: Explorer 筛选、统计和导出
**Reason**: Explorer 筛选和导出依赖已退役互补 case 表。
**Migration**: 删除相关 UI 控件和导出 helper。

#### Scenario: Explorer 筛选退役
- **WHEN** 用户启动 Gradio viewer
- **THEN** 系统不再要求提供 weak modality、case type、bucket 或 gain 筛选控件

### Requirement: Explorer 样本详情联动
**Reason**: 详情联动只服务于 Complementarity Explorer。
**Migration**: 保留普通 manifest 样本详情与 diagnostics 展示。

#### Scenario: Explorer 详情联动退役
- **WHEN** 用户在普通 viewer 中浏览样本
- **THEN** 系统不再要求根据 complementarity case 定位样本

### Requirement: 自动化测试
**Reason**: 互补分析后端和 Explorer 筛选逻辑已退役。
**Migration**: 删除对应测试，保留 viewer 和 manifest 的现有测试。

#### Scenario: 互补分析测试退役
- **WHEN** 开发者运行定向测试
- **THEN** 系统不再要求运行 `tests/test_complementarity_analysis.py` 或 `tests/test_gradio_complementarity_explorer.py`

### Requirement: 强势模态互补分析维度
**Reason**: strong/weak pair mode 是互补分析扩展，随能力一起退役。
**Migration**: 不保留 pair mode 入口。

#### Scenario: strong modality pair mode 退役
- **WHEN** 用户传入 strong modality 分析需求
- **THEN** 系统不再要求生成 strong/weak pair case

### Requirement: 强势模态预测来源与 case schema
**Reason**: strong modality case schema 属于已退役互补 case 输出。
**Migration**: 删除相关 schema 字段要求。

#### Scenario: strong modality case schema 退役
- **WHEN** 用户查看输出 schema
- **THEN** 系统不再要求 `strong_modality`、`strong_prediction_source` 或 `strong_weak_pair` 字段

### Requirement: 强弱融合 subset 可选降级
**Reason**: pair fusion subset 降级只服务于已退役 strong/weak case mining。
**Migration**: 不提供兼容降级语义。

#### Scenario: pair fusion subset 降级退役
- **WHEN** pair fusion prediction 缺失
- **THEN** 系统不再要求输出互补 case 的降级字段

### Requirement: 按强势模态汇总互补指标
**Reason**: 按 strong modality 和 pair 的互补 summary 已退役。
**Migration**: 不提供迁移。

#### Scenario: strong modality summary 退役
- **WHEN** 用户查看 summary
- **THEN** 系统不再要求包含 `by_strong_modality` 或 `by_strong_weak_pair`

### Requirement: Explorer 强势模态筛选
**Reason**: Strong Modality 控件属于已退役 Explorer。
**Migration**: 删除对应 viewer 控件。

#### Scenario: Strong Modality 控件退役
- **WHEN** 用户启动 viewer
- **THEN** 系统不再要求显示 Strong Modality 筛选控件

### Requirement: Explorer 强势模态样本详情
**Reason**: strong modality 详情展示依赖已退役 pair mode。
**Migration**: 保留普通样本详情。

#### Scenario: strong modality 详情退役
- **WHEN** 用户选择 viewer 样本
- **THEN** 系统不再要求显示 strong/weak pair prediction metadata

### Requirement: 强势模态互补自动化测试
**Reason**: strong modality 互补分析已退役。
**Migration**: 删除对应测试。

#### Scenario: strong modality 互补测试退役
- **WHEN** 开发者运行测试
- **THEN** 系统不再要求覆盖 strong modality case mining 或 Explorer strong modality 筛选
