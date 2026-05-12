## Context

现有项目已经具备两个可复用基础：

- Conditional Utility Audit 会写出 `subset_predictions.csv.gz`、`conditional_utility_per_sample_delta.csv.gz`、`conditional_utility_by_bucket.csv`、`teacher_predictions.csv.gz` 等产物。当前 Scene32 的 `subset_predictions.csv.gz` 字段包含 `sample_id`、`dataset_index`、`scene_id`、`scene_slug`、`split`、原始模态路径、`horizon_idx`、`horizon_name`、`gt_beam`、`subset_name`、`gt_prob`、`ce`、`top1_hit`、`pred_top1..5`、`top1_prob..5` 等；它没有完整 64 类概率分布。当前单弱模态预测在 `teacher_predictions.csv.gz` 中可用，`subset_name` / `teacher_modality` 覆盖 `image`、`radar`、`lidar`。
- Gradio viewer 已经能从 manifest 展示 raw / processed image、LiDAR、radar、GPS、mmWave，并在 Diagnostics 区展示预测、置信度和 future beam distribution。它目前以单一页面布局组织控件和三列展示，筛选逻辑位于 `tools/visualization/viewer_utils.py`，渲染入口位于 `tools/visualization/gradio_multimodal_viewer.py`。

本变更要做的是消费已有 audit 和 viewer 产物，不重新运行训练或模型推理。分析目标是把“弱模态是否存在局部互补信息”拆解为可筛选、可导出、可点开查看原始样本的证据表。

## Goals / Non-Goals

**Goals:**

- 生成每个 `sample_id × weak_modality × horizon` 的互补 case 表，覆盖 strong-only、weak-only / weak-teacher、strong-plus-weak fusion 的 top1 预测、正确性、case type 和可用概率指标。
- 输出全局、按弱模态、按 bucket 的核心研究指标：`complementarity_rate`、`rescue_rate_given_complementary`、`unused_complementary_rate`、`negative_transfer_rate`、`net_fusion_gain_count`。
- 在概率字段可用时计算 `p_true_*`、`*_margin`、`weak_gt_gain`、`fusion_gt_gain`；不可用时保留 top1 case mining 并明确标记概率指标不可用。
- 在现有 Gradio viewer 中新增 Complementarity Explorer Tab，支持筛选、排序、统计、图表、表格、导出和点击样本联动现有样本详情。
- 让后端分析和前端筛选逻辑都可以单元测试，不依赖真实 Scene32 全量数据才能验证核心行为。

**Non-Goals:**

- 不修改 MARF、CRAF、G2D 或 teacher 模型结构。
- 不改变 Conditional Utility Audit 已有输出字段含义，也不要求重新跑 audit 才能使用已有产物。
- 不在本阶段实现 JS divergence、KL divergence 或新的训练时 safe fusion gate；这些只作为后续研究输入。
- 不保证在缺少 manifest 或模态资源文件时仍能展示完整原始五模态；此时 Explorer 必须显示 case 表和安全空状态。

## Decisions

### Decision 1: 新增独立分析模块，CLI 只做离线产物构建

实现时新增 `scripts/analysis/build_complementarity_cases.py` 作为命令入口，并把可测试逻辑放入 `src/kd_sensing/diagnostics/complementarity.py` 或同级模块。脚本职责是读取表、调用 schema adapter、构建 case 表、写出 summary / bucket / report。

理由：互补分析是 audit 产物的消费者，和模型 forward 解耦后可以快速迭代，也能在没有 GPU 的环境中运行。

替代方案是把逻辑并入 `run_conditional_utility_audit.py`。该方案可以少一次命令，但会让 audit runner 继续膨胀，并把研究解释逻辑和推理流程耦合，不利于复用已有产物。

### Decision 2: schema adapter 统一字段与 subset 命名

后端先把输入表规范化为内部列：

- 样本键：`sample_id`、`dataset_index`、`scene`、`horizon_idx`、`horizon_name`
- 标签与预测：`y_true`、`subset_name`、`pred_top1`、`pred_top2`、`top1_prob`、`top2_prob`、`p_true`
- 元数据：原始索引字段、模态路径、scene / split、bucket 字段

强势路径默认解析为 `strong_only`，weak-plus-fusion 默认解析为 `strong_plus_<weak>`，同时兼容 `gps+mmwave`、`gps+mmwave+image` 等显示命名。弱模态预测优先从以下来源解析：

1. `subset_predictions` 中的单弱模态 subset，例如 `image`、`radar`、`lidar` 或配置别名；
2. `teacher_predictions` 中的 `teacher_modality` / `subset_name`；
3. 显式配置的 weak prediction table；
4. 均不可用时将 `weak_prediction_available=false`，仅输出不依赖 weak correctness 的 fusion case，并在 summary 和 UI 中标记限制。

理由：当前实际产物的 weak-only 单模态预测在 `teacher_predictions.csv.gz` 中，而用户文档优先列出的 `subset_predictions.csv.gz` 主要覆盖 strong-only 和 strong-plus-weak。adapter 能避免把当前字段假设写死。

替代方案是强制要求 `subset_predictions` 必须包含单弱模态预测。该方案实现简单，但会让当前已有 Scene32 产物不可直接使用。

### Decision 3: case type 同时输出研究标签和互斥主类

每行输出布尔字段：

- `strong_correct = strong_pred == y_true`
- `weak_correct = weak_pred == y_true`
- `fusion_correct = fusion_pred == y_true`
- `is_potential_complementary = strong_wrong && weak_correct`
- `is_rescue = strong_wrong && weak_correct && fusion_correct`
- `is_unused_complementary = strong_wrong && weak_correct && fusion_wrong`
- `is_fusion_rescue = strong_wrong && fusion_correct`
- `is_negative_transfer = strong_correct && fusion_wrong`

同时输出 `case_type` 作为互斥主类，优先级为 rescue、unused complementary、negative transfer、strong_wrong_fusion_correct、all_correct、all_wrong、other；再输出 `research_tags` 作为可多选标签，允许表达 B/C 是 A 的子集。

理由：研究解释需要承认集合包含关系，前端又需要避免重复计数。互斥主类用于条形图和汇总，研究标签用于筛选潜在互补样本。

替代方案是只输出单个 case type。该方案容易把 `strong_wrong_weak_correct` 与 rescue / unused 重复计数，导致前端解释不清。

### Decision 4: 概率指标按可用字段渐进增强

当前 `subset_predictions.csv.gz` 提供 `gt_prob`、`top1_prob`、`top2_prob`，因此可以计算：

- `p_true_strong`、`p_true_weak`、`p_true_fusion`
- `strong_margin`、`weak_margin`、`fusion_margin`
- `weak_gt_gain = p_true_weak - p_true_strong`
- `fusion_gt_gain = p_true_fusion - p_true_strong`

如果未来输入提供完整 logits / probability vector，再扩展 JS divergence / KL divergence 和 64 类分布导出。当前后端不从 top-k 字段伪造完整分布。

理由：避免用不完整 top-k 字段制造看似完整的概率分布。样本详情中的 64 类分布继续复用 viewer manifest 的 `beam_distribution`，若 manifest 没有完整分布则显示 top-k 表和空状态。

替代方案是把 top-k 概率补零成 64 类分布。该方案会产生错误的分布距离和误导性图表。

### Decision 5: Gradio Explorer 复用现有样本渲染，新增独立筛选状态

Viewer 新增 `--complementarity-dir` 参数或自动探测 `outputs/{scene}/complementarity_analysis/`。当目录可用时，页面展示 Complementarity Explorer Tab；当不可用时显示说明和空表，不影响原有 viewer。

Explorer 的筛选、排序、统计和导出逻辑放入可测试 helper，返回：

- 当前筛选 case 表；
- 指标摘要；
- case type 条形图；
- bucket rate 图；
- 选中行对应的 `sample_id` / `dataset_index`。

点击 case 表行后，通过 `sample_id` 或 `dataset_index` 定位现有 manifest 样本，并调用已有 `render_sample()` / 静态渲染逻辑更新样本详情区。定位失败时保留 case 详情 JSON，并显示模态资源不可用。

理由：现有 viewer 已经解决了路径解析、图片/Plotly 渲染和 future distribution 空状态；Explorer 只需要新增研究表和导航到样本的桥接逻辑。

替代方案是另建一个 Gradio app。该方案隔离性强，但用户需要在两个页面之间切换，不能直接从 case 表查看原始样本。

## Risks / Trade-offs

- [Risk] 弱模态预测来源不统一，可能来自 teacher 而不是和 strong-plus 同一个 fusion checkpoint → Mitigation：输出 `weak_prediction_source`，summary/report 明确记录来源，UI 显示来源；当来源缺失时禁用依赖 weak correctness 的指标。
- [Risk] `sample_id` 在不同产物或 manifest 中不完全一致 → Mitigation：使用 `sample_id + dataset_index + horizon_idx` 作为 join key 优先级，并保留 `dataset_index` fallback；无法匹配时输出 unmatched 计数。
- [Risk] bucket 表是聚合产物，可能缺少逐样本 bucket 分配 → Mitigation：优先合并 `communication_state_features.csv.gz` 的逐样本 bucket 字段；只有聚合 bucket 表时只输出 `complementarity_by_bucket.csv` 的可比聚合或记录不可用原因。
- [Risk] Gradio Dataframe 行选择在不同版本中事件 payload 有差异 → Mitigation：把行选择逻辑封装成小函数并测试，必要时提供 `sample_id` 输入框作为 fallback。
- [Risk] case 表较大导致前端加载慢 → Mitigation：后端预写压缩 CSV，前端按筛选返回限定行数和总数，导出时写出完整 filtered CSV；图表基于筛选后的 DataFrame 聚合。
- [Risk] 研究标签和互斥 case 容易被误读 → Mitigation：UI 固定显示简短解释：潜在互补、rescue、unused complementary、negative transfer 的含义，并在 summary 分别列出分母。

## Migration Plan

1. 新增后端模块与脚本，先对当前 Scene32 产物运行，确认 schema mapping、行数和输出文件。
2. 新增 Gradio 参数和 Explorer Tab，在没有 complementarity 输出时保持原 viewer 行为不变。
3. 增加测试并用 `conda run -n kd_mm_beam pytest ...` 验证核心逻辑。
4. 如果需要回滚，只需不传 `--complementarity-dir` 或删除 `outputs/{scene}/complementarity_analysis/`；训练与 audit 产物不受影响。

## Open Questions

- `weak_pred` 的研究语义是否固定使用单模态 teacher，还是后续要训练真实 weak-only student / fusion subset。当前方案通过 `weak_prediction_source` 记录来源，并允许配置覆盖。
- 多 horizon 对比在 P0 中按单 horizon 筛选展示；是否需要在 P2 做同一 sample 的跨 horizon 对比视图。
- bucket 的逐样本分配如果当前产物不足，是否要从 `communication_state_features.csv.gz` 重新生成统一 bucket 列，还是只保留聚合 bucket 对比。
