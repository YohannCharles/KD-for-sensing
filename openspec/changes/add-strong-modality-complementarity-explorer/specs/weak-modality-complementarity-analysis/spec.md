## ADDED Requirements

### Requirement: 强势模态互补分析维度
互补分析后端 MUST 支持以一个或多个强势模态作为 anchor 生成互补 case。强势模态维度 MUST 与现有弱模态维度正交，输出粒度 MUST 至少达到 `sample_id × strong_modality × weak_modality × horizon`。

#### Scenario: 指定单个强势模态和全部弱模态
- **WHEN** 用户运行互补分析命令并指定 `--strong-modalities mmwave --weak-modalities image,radar,lidar`
- **THEN** 系统 MUST 为 `mmwave × image`、`mmwave × radar` 和 `mmwave × lidar` 生成逐样本逐 horizon case
- **AND** `complementarity_cases.csv.gz` MUST 包含 `strong_modality=mmwave`
- **AND** summary metadata MUST 记录实际采用的 strong modalities 和 weak modalities

#### Scenario: 默认强势模态集合
- **WHEN** 用户没有显式指定 `--strong-modalities`
- **THEN** 系统 MUST 使用配置或默认策略选择强势模态集合
- **AND** 对当前 Scene32 默认 MUST 支持 `gps` 和 `mmwave`
- **AND** summary metadata MUST 记录默认值来源

#### Scenario: 强势模态预测来源缺失
- **WHEN** 某个 requested strong modality 在 `subset_predictions` 和 `teacher_predictions` 中都不可用
- **THEN** 系统 MUST 跳过该 strong modality 的 case 生成
- **AND** summary metadata MUST 在 warnings 或 unavailable sources 中记录缺失原因
- **AND** 其他可用 strong modality 的分析 MUST 继续完成

### Requirement: 强势模态预测来源与 case schema
系统 MUST 为每个 strong/weak 组合记录强势模态、弱模态、预测来源、预测值、正确性和可用概率字段。系统 MUST 保留现有 `strong_only` 默认分析行为，并在启用强势模态维度时新增字段而不是破坏原有字段含义。

#### Scenario: 使用 teacher predictions 作为强势模态来源
- **WHEN** `teacher_predictions` 包含 `teacher_modality=mmwave` 的预测行
- **THEN** 系统 MUST 能将其作为 `mmwave` 的 strong prediction 来源
- **AND** case 表 MUST 包含 `strong_prediction_source=teacher_predictions`
- **AND** case 表 MUST 包含 `strong_pred`、`strong_correct`、`p_true_strong` 和 `strong_margin`

#### Scenario: 使用单模态 subset 作为强势模态来源
- **WHEN** `subset_predictions` 包含能解析为 requested strong modality 的单模态 subset
- **THEN** 系统 MUST 优先或按配置使用该 subset 作为 strong prediction 来源
- **AND** summary metadata MUST 记录该 strong modality 的 source、row count 和 resolved subset name

#### Scenario: 输出 strong weak pair 字段
- **WHEN** case 表成功生成
- **THEN** `complementarity_cases.csv.gz` MUST 包含 `strong_modality`、`weak_modality`、`strong_prediction_source`、`weak_prediction_source` 和 `strong_weak_pair`
- **AND** `strong_weak_pair` MUST 能唯一表达当前行的 strong/weak 组合，例如 `mmwave+image`

### Requirement: 强弱融合 subset 可选降级
系统 MUST 支持为 strong/weak pair 配置可选 fusion subset。若某个 strong/weak pair 没有对应 fusion prediction，系统 MUST 仍输出 strong-vs-weak 互补关系，并将 fusion 相关字段和 rescue 指标安全降级为空值或不可用状态。

#### Scenario: strong plus weak fusion subset 可用
- **WHEN** `subset_predictions` 包含 requested pair 对应的 fusion subset
- **THEN** 系统 MUST 读取该 fusion prediction
- **AND** case 表 MUST 设置 `fusion_prediction_available=true`
- **AND** case 表 MUST 输出 `fusion_pred`、`fusion_correct`、`p_true_fusion`、`fusion_gt_gain` 和现有 rescue / unused complementary / negative transfer 标签

#### Scenario: strong plus weak fusion subset 缺失
- **WHEN** requested pair 没有可用 fusion subset
- **THEN** 系统 MUST 设置 `fusion_prediction_available=false`
- **AND** case 表 MUST 保留 `strong_correct`、`weak_correct`、`strong_wrong_weak_correct` 相关标签和强弱概率差
- **AND** rescue、unused complementary、negative transfer、`fusion_pred` 和 `fusion_gt_gain` MUST 输出为空值或标记不可用
- **AND** summary metadata MUST 记录缺失 fusion subset 的 strong/weak pair

#### Scenario: 保留原 strong_only plus weak 行为
- **WHEN** 用户按旧方式只指定 `--strong-subset strong_only` 且不启用 strong modality 维度
- **THEN** 系统 MUST 保持现有 `strong_only` 与 `strong_plus_<weak>` 分析语义
- **AND** 既有输出字段、默认 weak modality 和 summary 指标 MUST 保持兼容

### Requirement: 按强势模态汇总互补指标
系统 MUST 在 summary 中输出按 strong modality 和 strong/weak pair 分组的互补指标。所有比例 MUST 显式记录分子、分母和值，并在分母为 0 或指标不可用时输出空值和不可用原因。

#### Scenario: 输出按 strong modality 的 summary
- **WHEN** case 表包含 `strong_modality`
- **THEN** `complementarity_summary.json` MUST 包含 `by_strong_modality`
- **AND** 每个 strong modality 的 summary MUST 包含 `complementarity_rate`、`strong_wrong_count`、`complementary_count` 和可用 fusion 指标

#### Scenario: 输出按 strong weak pair 的 summary
- **WHEN** case 表包含多个 strong/weak 组合
- **THEN** `complementarity_summary.json` MUST 包含 `by_strong_weak_pair`
- **AND** 每个 pair 的 summary MUST 至少包含 sample count、horizon count、complementarity rate、mean weak gt gain 和 fusion availability

#### Scenario: fusion 指标不可用时降级
- **WHEN** 某个 strong/weak pair 没有 fusion prediction
- **THEN** 对应 pair 的 rescue、unused complementary、negative transfer 和 fusion gain 指标 MUST 标记为不可用
- **AND** strong/weak top1 互补率 MUST 继续输出

### Requirement: Explorer 强势模态筛选
Complementarity Explorer MUST 提供 `Strong Modality` 控件。该控件 MUST 与现有 `Weak Modality` 控件一致支持单值和 `all`，并 MUST 参与统计、图表、表格、样本详情和导出。

#### Scenario: 加载 strong modality choices
- **WHEN** Explorer 加载的 case 表包含 `strong_modality`
- **THEN** 页面 MUST 显示 `Strong Modality` 控件
- **AND** choices MUST 包含 `all` 和 case 表中可用的 strong modalities
- **AND** 默认值 MUST 优先使用 `mmwave`，否则使用第一个可用 strong modality

#### Scenario: 应用 strong modality 筛选
- **WHEN** 用户选择 `Strong Modality=mmwave` 并点击 Apply filters
- **THEN** Explorer MUST 只展示 `strong_modality=mmwave` 的 case
- **AND** Filtered Statistics、Case Type Counts、Bucket Counts 和样本表 MUST 同步更新

#### Scenario: 查看某个强势模态与全部弱模态
- **WHEN** 用户选择一个 strong modality 且 `Weak Modality=all`
- **THEN** Explorer MUST 展示该 strong modality 与所有可用 weak modalities 的 case
- **AND** 表格 MUST 保留 `strong_modality`、`weak_modality` 和 `strong_weak_pair` 字段

#### Scenario: 查看全部强弱组合
- **WHEN** 用户选择 `Strong Modality=all` 且 `Weak Modality=all`
- **THEN** Explorer MUST 展示所有 strong/weak pair 的 case
- **AND** 统计 MUST 基于当前筛选后的全部 pair 计算

#### Scenario: 导出强势模态筛选结果
- **WHEN** 用户点击 Export filtered CSV
- **THEN** 导出文件 MUST 包含当前筛选结果的全部 case 字段
- **AND** 导出文件 MUST 包含 `strong_modality`、`strong_prediction_source` 和 `strong_weak_pair`

### Requirement: Explorer 强势模态样本详情
Explorer 在用户选择 case 表行时 MUST 在详情区展示 strong modality anchor 的预测信息，并继续复用现有样本定位和 raw / processed / diagnostics 展示能力。

#### Scenario: 点击强弱互补 case 行
- **WHEN** 用户在包含 `strong_modality` 的 case 表中选择一行
- **THEN** 详情 JSON MUST 展示 `strong_modality`、`weak_modality`、`strong_prediction_source`、`weak_prediction_source`、`strong_pred`、`weak_pred`、`fusion_pred` 和可用概率
- **AND** viewer MUST 尝试通过 `sample_id` 或 `dataset_index` 定位 manifest 样本

#### Scenario: fusion prediction 不可用的详情展示
- **WHEN** 选中 case 的 `fusion_prediction_available=false`
- **THEN** 详情 JSON MUST 明确标记 fusion prediction unavailable
- **AND** raw / processed modalities 和现有 diagnostics 区域 MUST 继续按 manifest 可用性展示或安全空状态

### Requirement: 强势模态互补自动化测试
系统 MUST 为强势模态互补分析和 Explorer 筛选提供自动化测试。测试 MUST 使用小型人工 DataFrame 验证核心语义，不依赖真实 Scene32 全量输出。

#### Scenario: strong modality case mining 测试
- **WHEN** 测试构造包含 `gps`、`mmwave`、`image`、`radar`、`lidar` teacher predictions 的小型输入
- **THEN** case mining MUST 能为指定 strong modality 和多个 weak modalities 输出预期行
- **AND** 输出 MUST 包含正确的 `strong_modality`、`strong_weak_pair` 和 prediction source

#### Scenario: 缺失 fusion subset 降级测试
- **WHEN** 测试输入缺少 strong plus weak fusion subset
- **THEN** case mining MUST 保留 strong/weak 互补 case
- **AND** fusion 相关字段和 summary 指标 MUST 按不可用语义输出

#### Scenario: Explorer strong modality 筛选测试
- **WHEN** 测试调用 Explorer 筛选 helper 并指定 strong modality、weak modality、case type、horizon 和 sort
- **THEN** helper MUST 返回预期 strong/weak pair 行集合、排序顺序和统计摘要
