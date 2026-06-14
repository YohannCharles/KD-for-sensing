# deepsense6g-gps-top8-candidate-selector Specification

## Purpose
定义 DeepSense6G GPS TopK candidate manifest、dataset 字段和 loss/metric 支撑契约。该 capability 为 BGAM 或其它当前 workflow 提供 supporting-only 候选语义；旧 DeepSense6G GPS Top8 selector 训练、plot、compare workflow 已退役，不属于当前 standalone 入口。
## Requirements
### Requirement: Top8 candidate manifest 支撑语义
BGAM 或其它当前 workflow MAY 通过包内 helper 从 GPS v2 logits 重新计算 Top8 candidates，并将 GPS context、candidate metadata、Top8 hit/miss label、nearest/oracle candidate、optional modality availability 和对齐诊断写为每样本一行。系统 MUST 不为旧 selector 恢复 standalone manifest CLI、root config 或默认输出 workflow。

#### Scenario: 使用真实 GPS logits 构建 Top8
- **WHEN** 用户运行 Top8 candidate manifest builder 且 `topk=8`
- **THEN** 系统 MUST 优先读取 `gps_logits.npy`、`logits.npy` 或 `pred_logits.npy`
- **AND** 系统 MUST 使用 `gps_logits_index.csv` 或等价 index 将 logits 与 prediction/sample 对齐
- **AND** Top8 candidates MUST 从 GPS v2 logits 重新计算
- **AND** 系统 MUST NOT 从 `predictions.csv` 中已有 top5 或截断列硬推 Top8

#### Scenario: GPS logits 缺失时早失败
- **WHEN** GPS v2 r15 结果目录缺少 logits 或 logits index
- **THEN** manifest builder MUST 报错
- **AND** 错误信息 MUST 指出需要先重跑 GPS v2 并保存 logits
- **AND** 本 workflow MUST NOT fallback 到 Gaussian top1 prior 生成 Top8 candidates

#### Scenario: manifest 字段完整
- **WHEN** manifest builder 写出 `top8_candidate_manifest.csv`
- **THEN** 每行 MUST 至少包含 scene、sample id、timestamp 或 frame id、support/query role、split role、target label、GPS top1、GPS top1/top2 prob、GPS top1-top2 margin、GPS entropy、GPS circular error、GPS signed residual、theta、range、E、N、heading、speed 可用值、image path、image_exists、camera AE row index、LiDAR feature path 和 radar feature path
- **AND** 每行 MUST 包含 `cand0_beam` 到 `cand7_beam`、`cand0_logit` 到 `cand7_logit`、`cand0_prob` 到 `cand7_prob`、`cand0_rank` 到 `cand7_rank`、`cand0_dist_to_gps_top1` 到 `cand7_dist_to_gps_top1`
- **AND** 每行 MUST 包含 `target_in_top8`、`target_candidate_index`、`nearest_candidate_index`、`nearest_candidate_error`、`top8_oracle_error`、`top8_oracle_beam` 和 `top8_miss`

#### Scenario: Top8 recall 与已有分析对齐
- **WHEN** manifest builder 完成 Top8 candidate 计算
- **THEN** 系统 MUST 计算 overall 与 by-scene Top8 recall
- **AND** 系统 MUST 与 `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep/topk_analysis/` 中已有 TopK analysis 进行对齐检查
- **AND** 若差异明显，系统 MUST 写出 warning 并记录到 metadata

### Requirement: TopK candidate dataset 支撑语义
保留的 TopK candidate dataset helper MAY 用于当前 BGAM 或候选重排支撑路径读取 Top8 candidate manifest 并返回稳定张量字段。dataset helper MUST 支持没有 optional modality 时运行 GPS context-only 支撑逻辑，并 MUST 只用 train/source/support 样本拟合 E/N/log_range/speed 等 normalization 参数；它不得恢复旧 selector 训练入口。

#### Scenario: dataset 返回候选与标签张量
- **WHEN** dataset 读取一个 Top8 manifest 样本
- **THEN** 返回样本 MUST 包含 `candidate_beams: LongTensor [8]`
- **AND** 返回样本 MUST 包含 `candidate_logits: FloatTensor [8]`、`candidate_probs: FloatTensor [8]` 和 `candidate_features: FloatTensor [8, F_cand]`
- **AND** 返回样本 MUST 包含 `gps_context: FloatTensor [D_gps]`
- **AND** 返回样本 MUST 包含 `target_label`、`target_in_top8`、`target_candidate_index`、`nearest_candidate_index`、`top8_oracle_error` 和 `miss_label`

#### Scenario: candidate feature 契约
- **WHEN** dataset 构造 `candidate_features`
- **THEN** 每个 candidate feature MUST 至少包含 beam sin/cos、rank norm、logit norm、prob、log prob、dist-to-GPS-top1 norm、is GPS top1、is GPS top3 和 is GPS top5
- **AND** circular distance MUST 按 `num_beams=64` 的 wrap-around 语义计算

#### Scenario: GPS context feature 契约
- **WHEN** dataset 构造 `gps_context`
- **THEN** GPS context MUST 至少包含 E/N norm、sin/cos theta、log range norm、sin/cos heading、speed norm、GPS top1/top2 prob、GPS margin、GPS entropy 和 GPS predicted beam sin/cos
- **AND** normalization fit MUST NOT 使用 target query label 或 target query 统计量

#### Scenario: optional modality 缺失时降级
- **WHEN** camera AE、image tensor、LiDAR feature 或 radar feature 不可用
- **THEN** dataset MUST 返回缺失标记或空字段
- **AND** GPS context-only selector MUST 仍可构建和运行
- **AND** 对应 optional ablation MUST 在 summary 中记录 `skipped_reason`

### Requirement: TopK candidate selector loss 支撑语义
保留的 `TopKCandidateSelectorLoss` 或等价 loss helper MAY 作为 BGAM/候选支撑代码使用。总 loss MUST 由 candidate circular soft CE、target-in-Top8 index CE、miss BCE、GPS prior anchor KL 和 entropy regularization 组成，并 MUST 支持 hard-rank sample weighting；该 loss helper 本身不得重新暴露旧 selector standalone workflow。

#### Scenario: candidate circular soft CE 对所有样本有效
- **WHEN** 系统计算 candidate soft target
- **THEN** soft target MUST 按 candidate beam 与 target label 的 circular distance 构造
- **AND** 权重 MUST 与 `exp(-distance^2 / (2 * sigma^2))` 成比例
- **AND** target 不在 Top8 时 nearest candidate MUST 获得最大 soft target 权重

#### Scenario: target index CE 只在 Top8 hit 上计算
- **WHEN** `target_in_top8 == 1`
- **THEN** 系统 MUST 使用 `target_candidate_index` 计算普通 CE
- **AND** `target_in_top8 == 0` 的样本 MUST 不参与 target index CE

#### Scenario: miss head 学习 Top8 miss
- **WHEN** 系统计算 miss loss
- **THEN** `miss_label` MUST 等于 `1` if target 不在 Top8 else `0`
- **AND** miss BCE MUST 只使用允许训练的 source/support 样本

#### Scenario: prior anchor 保护 GPS good 样本
- **WHEN** 样本的 GPS top1 circular error 小于 `good_error_threshold`
- **THEN** GPS prior anchor MUST 约束 selector candidate distribution 不要显著偏离 GPS candidate probability distribution
- **AND** target query 样本 MUST NOT 参与 prior anchor 训练

#### Scenario: hard-rank 样本加权
- **WHEN** `target_in_top8 == 1` 且 `target_candidate_index > 0`
- **THEN** 样本权重 MUST 增加 `hard_rank_weight` 或等价配置权重
- **AND** 默认 `hard_rank_weight` MUST 为 `2.0`

### Requirement: DeepSense6G GPS Top8 candidate selector 已退役
DeepSense6G GPS Top8 candidate selector 训练/plot/compare workflow 不再属于当前支持能力。系统 MUST 不再提供 selector/attention selector、runner、plotter、comparison report 或默认配置。BGAM 依赖的 TopK candidate manifest/dataset/loss 支撑代码 MAY 保留。

#### Scenario: Top8 selector 入口不存在
- **WHEN** 开发者检查 console scripts、配置和包内模块
- **THEN** 项目 MUST 不声明 DeepSense6G Top8 selector 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留 `configs/deepsense6g_top8_selector.yaml`
- **AND** 项目 MUST 不保留 Top8 selector 专属 model、engine 或 tests
