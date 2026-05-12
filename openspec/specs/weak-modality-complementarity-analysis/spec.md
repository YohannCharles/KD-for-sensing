# weak-modality-complementarity-analysis Specification

## Purpose
定义从 Conditional Utility Audit 产物生成弱模态互补样本分析结果，并在 Gradio viewer 中探索、筛选、导出互补 case 的能力。
## Requirements
### Requirement: 互补分析命令入口
系统 MUST 提供独立命令入口，用于从已有 Conditional Utility Audit 产物生成弱模态互补样本分析结果。该入口 MUST 不加载训练 checkpoint、不构建训练器、不执行模型 forward。

#### Scenario: 运行 Scene32 互补分析
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/analysis/build_complementarity_cases.py --scene scene32 --input-path outputs/scene32/marf/conditional_utility --output-dir outputs/scene32/complementarity_analysis`
- **THEN** 系统 MUST 读取 `subset_predictions` 产物
- **AND** 系统 MUST 在输出目录生成 `complementarity_cases.csv.gz`、`complementarity_summary.json` 和 `complementarity_report.md`
- **AND** 系统 MUST 在可获得 bucket 信息时生成 `complementarity_by_bucket.csv`

#### Scenario: 命令行参数覆盖默认 subset
- **WHEN** 用户通过 `--strong-subset`、`--weak-modalities`、`--fusion-subsets` 或 `--horizons` 指定分析范围
- **THEN** 系统 MUST 使用用户指定的 subset 与 horizon
- **AND** summary metadata MUST 记录实际采用的 strong subset、weak modalities、fusion subset mapping 和 horizon 列表

### Requirement: 输入 schema 适配
系统 MUST 在读取输入表后执行 schema adapter，基于真实字段生成内部标准 schema。Adapter MUST 兼容 csv、csv.gz 和 parquet，并 MUST 记录输入字段、字段映射、subset 映射和缺失能力。

#### Scenario: 适配当前 subset predictions 字段
- **WHEN** 输入表包含 `gt_beam`、`pred_top1`、`gt_prob`、`top1_prob`、`top2_prob`、`horizon_idx`、`horizon_name` 和 `subset_name`
- **THEN** adapter MUST 将 `gt_beam` 映射为 `y_true`
- **AND** adapter MUST 将 `pred_top1` 映射为对应 subset 的 top1 预测
- **AND** adapter MUST 将 `gt_prob` 映射为对应 subset 的真值概率
- **AND** adapter MUST 将 `top1_prob - top2_prob` 映射为对应 subset 的 margin

#### Scenario: 使用 teacher predictions 作为弱模态预测来源
- **WHEN** `subset_predictions` 不包含单弱模态 subset 但 `teacher_predictions` 包含 `image`、`radar` 或 `lidar`
- **THEN** 系统 MUST 使用 `teacher_predictions` 作为对应弱模态的 `weak_pred` 来源
- **AND** `complementarity_cases.csv.gz` MUST 包含 `weak_prediction_source`
- **AND** summary metadata MUST 记录该弱模态使用 teacher prediction 来源

#### Scenario: 概率字段缺失时降级
- **WHEN** 输入表缺少 `gt_prob`、`top1_prob` 或 `top2_prob`
- **THEN** 系统 MUST 继续基于 top1 prediction 和 y_true 生成 case mining 结果
- **AND** 概率相关字段 MUST 输出为空值
- **AND** summary metadata MUST 标记 `probability_metrics_available=false`

### Requirement: 逐样本 case 表
系统 MUST 为每个 `sample_id × weak_modality × horizon` 输出一行互补 case。Case 表 MUST 包含 strong-only、weak-only 或 weak-teacher、strong-plus-weak fusion 三类预测的 top1 结果与正确性。

#### Scenario: 写出 case 表核心字段
- **WHEN** strong、weak 和 fusion 三类预测都可对齐
- **THEN** `complementarity_cases.csv.gz` MUST 包含 `sample_id`、`dataset_index`、`scene`、`horizon_idx`、`horizon_name`、`weak_modality`、`y_true`、`strong_pred`、`weak_pred`、`fusion_pred`、`strong_correct`、`weak_correct`、`fusion_correct`、`case_type` 和 `research_tags`
- **AND** 输出 MUST 保留可用的 split、原始模态路径、root csv、原始索引字段和 bucket 字段

#### Scenario: 生成概率与增益字段
- **WHEN** strong、weak 和 fusion 的真值概率或 top1/top2 概率可用
- **THEN** case 表 MUST 包含 `p_true_strong`、`p_true_weak`、`p_true_fusion`、`weak_gt_gain`、`fusion_gt_gain`、`strong_margin`、`weak_margin` 和 `fusion_margin`
- **AND** `weak_gt_gain` MUST 等于 `p_true_weak - p_true_strong`
- **AND** `fusion_gt_gain` MUST 等于 `p_true_fusion - p_true_strong`

#### Scenario: 合并 per-sample delta
- **WHEN** 输入目录包含 `conditional_utility_per_sample_delta` 产物
- **THEN** 系统 MUST 按 `sample_id`、`dataset_index`、`horizon_idx`、`horizon_name` 和 `weak_modality` 合并可用 delta 字段
- **AND** case 表 MUST 保留 `delta_ce`、`delta_top1`、`delta_top3` 和 `delta_dba` 等已有字段

### Requirement: case 判定语义
系统 MUST 按 top1 prediction 判定 strong、weak 和 fusion 的正确性，并 MUST 同时输出互斥主类 `case_type` 与可多标签 `research_tags`。系统 MUST 避免把子集标签重复计入互斥统计。

#### Scenario: 判定潜在互补样本
- **WHEN** `strong_correct=false` 且 `weak_correct=true`
- **THEN** `research_tags` MUST 包含 `strong_wrong_weak_correct`
- **AND** 该样本 MUST 计入 `complementary_count`

#### Scenario: 判定 rescue 样本
- **WHEN** `strong_correct=false` 且 `weak_correct=true` 且 `fusion_correct=true`
- **THEN** `case_type` MUST 为 `strong_wrong_weak_correct_fusion_correct`
- **AND** `research_tags` MUST 同时包含 `strong_wrong_weak_correct` 和 `rescue`

#### Scenario: 判定 unused complementary 样本
- **WHEN** `strong_correct=false` 且 `weak_correct=true` 且 `fusion_correct=false`
- **THEN** `case_type` MUST 为 `strong_wrong_weak_correct_fusion_wrong`
- **AND** `research_tags` MUST 同时包含 `strong_wrong_weak_correct` 和 `unused_complementary`

#### Scenario: 判定负迁移样本
- **WHEN** `strong_correct=true` 且 `fusion_correct=false`
- **THEN** `case_type` MUST 为 `strong_correct_fusion_wrong`
- **AND** `research_tags` MUST 包含 `negative_transfer`

#### Scenario: 判定融合纠错但弱模态 top1 未必正确
- **WHEN** `strong_correct=false` 且 `fusion_correct=true`
- **THEN** `research_tags` MUST 包含 `strong_wrong_fusion_correct`
- **AND** 若 `weak_correct=false`，`case_type` MUST 不得被标记为 `strong_wrong_weak_correct_fusion_correct`

### Requirement: summary 指标
系统 MUST 输出全局、按 weak modality、按 horizon 的 summary 指标。所有比例 MUST 显式记录分子、分母和值，并在分母为 0 时输出空值而不是抛出异常。

#### Scenario: 计算核心研究指标
- **WHEN** case 表生成完成
- **THEN** `complementarity_summary.json` MUST 包含 `complementarity_rate`
- **AND** `complementarity_rate` MUST 等于 `count(strong_wrong_weak_correct) / count(strong_wrong)`
- **AND** summary MUST 包含 `rescue_rate_given_complementary`、`unused_complementary_rate`、`negative_transfer_rate` 和 `net_fusion_gain_count`
- **AND** `rescue_rate_given_complementary` MUST 等于 `count(strong_wrong_weak_correct_fusion_correct) / count(strong_wrong_weak_correct)`
- **AND** `unused_complementary_rate` MUST 等于 `count(strong_wrong_weak_correct_fusion_wrong) / count(strong_wrong_weak_correct)`
- **AND** `negative_transfer_rate` MUST 等于 `count(strong_correct_fusion_wrong) / count(strong_correct)`
- **AND** `net_fusion_gain_count` MUST 等于 `count(strong_wrong_fusion_correct) - count(strong_correct_fusion_wrong)`

#### Scenario: 汇总概率增益
- **WHEN** probability metrics 可用
- **THEN** summary MUST 包含全局、按 weak modality、按 case type 的 `mean_weak_gt_gain` 和 `mean_fusion_gt_gain`
- **AND** 当 probability metrics 不可用时，summary MUST 保留这些字段为空值并记录不可用原因

### Requirement: bucket 统计与报告
系统 MUST 在存在逐样本 bucket 或可映射 bucket 信息时输出按 bucket 的互补统计，并 MUST 生成一个模板化 markdown 报告解释关键结果。

#### Scenario: 生成 bucket 统计
- **WHEN** case 表包含 bucket 字段或输入目录包含可合并的 `communication_state_features` 字段
- **THEN** 系统 MUST 生成 `complementarity_by_bucket.csv`
- **AND** 输出 MUST 至少包含 `bucket_feature`、`bucket_name`、`weak_modality`、`horizon_name`、`sample_count`、`strong_wrong_count`、`strong_wrong_weak_correct_count`、`rescue_count`、`unused_complementary_count`、`negative_transfer_count` 及对应比例

#### Scenario: bucket 不可用时安全降级
- **WHEN** 输入产物无法提供逐样本 bucket 信息
- **THEN** 系统 MUST 仍生成 case 表、summary 和 report
- **AND** 系统 MUST 在 summary metadata 和 report 中记录 bucket statistics unavailable 的原因

#### Scenario: 生成模板化报告
- **WHEN** summary 生成完成
- **THEN** `complementarity_report.md` MUST 概述互补样本是否存在、互补率最高的弱模态、rescue 是否充分、负迁移是否显著、bucket high level 发现和概率指标可用性

### Requirement: Complementarity Explorer 页面
系统 MUST 在现有 Gradio viewer 中提供 `Complementarity Explorer` / `弱模态互补样本分析` Tab。该 Tab MUST 能在 complementarity 输出缺失时显示空状态，并 MUST 不影响原有样本浏览、raw/processed modalities 和 diagnostics 功能。

#### Scenario: 加载互补分析目录
- **WHEN** 用户使用 `--complementarity-dir outputs/scene32/complementarity_analysis` 启动 viewer
- **THEN** viewer MUST 加载 `complementarity_cases.csv.gz` 和 `complementarity_summary.json`
- **AND** 页面 MUST 显示 Complementarity Explorer Tab
- **AND** 原有 viewer 控件和样本浏览 MUST 保持可用

#### Scenario: 输出缺失时显示空状态
- **WHEN** 用户没有提供 complementarity 目录或目录中缺少 case 表
- **THEN** Complementarity Explorer MUST 显示空状态
- **AND** Gradio 页面 MUST 不因缺失文件抛出未处理异常

### Requirement: Explorer 筛选、统计和导出
Complementarity Explorer MUST 支持按 scene、horizon、weak modality、case type、bucket、排序字段和最小增益阈值筛选 case。页面 MUST 显示当前筛选结果的关键统计、图表和表格，并 MUST 支持导出 filtered CSV。

#### Scenario: 默认筛选
- **WHEN** Explorer 首次加载且存在 Scene32 case 表
- **THEN** 默认 scene MUST 为 `scene32`
- **AND** 默认 horizon MUST 为 `t+1`
- **AND** 默认 weak modality MUST 为 `image`
- **AND** 默认 case type MUST 包含 `strong_wrong_weak_correct`、`strong_wrong_weak_correct_fusion_correct`、`strong_wrong_weak_correct_fusion_wrong` 和 `strong_correct_fusion_wrong`

#### Scenario: 应用筛选并排序
- **WHEN** 用户修改 weak modality、case type、bucket 或增益阈值并点击 Apply filters
- **THEN** Explorer MUST 更新统计面板、case type 图、bucket 图和样本表
- **AND** 若 probability metrics 可用且用户选择 `weak_gt_gain desc`，样本表 MUST 按 `weak_gt_gain` 降序排列
- **AND** 若排序字段不可用，系统 MUST 回退到 `sample_id` 排序并显示提示

#### Scenario: 导出筛选结果
- **WHEN** 用户点击 Export filtered CSV
- **THEN** 系统 MUST 写出当前筛选结果 CSV
- **AND** 导出文件 MUST 包含当前表格所依据的全部 case 字段

### Requirement: Explorer 样本详情联动
Complementarity Explorer MUST 在用户选择 case 表行时联动样本详情区。详情区 MUST 展示 case 解释、真实标签、strong/weak/fusion top1 与可用概率，并 MUST 尽量复用现有 viewer 的原始五模态、processed 五模态和 future beam distribution 展示。

#### Scenario: 点击 case 行展示样本详情
- **WHEN** 用户在 Explorer 表格中选择一行
- **THEN** viewer MUST 根据 `sample_id` 或 `dataset_index` 定位 manifest 样本
- **AND** 详情区 MUST 展示该样本的 raw modalities、processed modalities、label、strong prediction、weak prediction、fusion prediction 和 case tags

#### Scenario: manifest 样本不可匹配
- **WHEN** case 表行无法匹配到 viewer manifest 样本
- **THEN** Explorer MUST 仍显示 case metadata 和预测信息
- **AND** 原始模态与 processed 模态区域 MUST 显示安全空状态

#### Scenario: 完整分布不可用
- **WHEN** 选中样本没有完整 `beam_distribution` 概率或 logits
- **THEN** Explorer MUST 显示 top-k 预测和真值概率信息
- **AND** 64 类分布区域 MUST 显示 `probability distribution unavailable` 或等价空状态

### Requirement: 自动化测试
系统 MUST 为互补分析后端和 Explorer 筛选逻辑提供自动化测试。测试 MUST 能用小型人工 DataFrame 验证核心语义，不依赖真实 Scene32 全量输出。

#### Scenario: case 判定测试
- **WHEN** 测试构造覆盖 strong wrong weak correct、rescue、unused complementary、negative transfer、all correct、all wrong 和 other 的样本
- **THEN** case mining 函数 MUST 输出预期的 `case_type` 和 `research_tags`

#### Scenario: summary 指标测试
- **WHEN** 测试输入包含已知数量的 strong wrong、complementary、rescue、unused complementary 和 negative transfer 样本
- **THEN** summary 函数 MUST 输出预期的核心比例和 `net_fusion_gain_count`

#### Scenario: schema adapter 与缺失概率测试
- **WHEN** 测试输入使用 subset 别名或缺少概率字段
- **THEN** adapter MUST 解析到正确的 strong / weak / fusion 行
- **AND** case mining MUST 正常运行且概率字段为空值

#### Scenario: 前端筛选测试
- **WHEN** 测试调用 Explorer 筛选 helper 并指定 case type、weak modality、horizon、bucket 和排序字段
- **THEN** helper MUST 返回预期行集合、排序顺序和统计摘要

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

