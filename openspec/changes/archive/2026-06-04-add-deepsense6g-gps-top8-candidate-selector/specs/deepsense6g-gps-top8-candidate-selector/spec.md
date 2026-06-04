## ADDED Requirements

### Requirement: DeepSense6G GPS Top8 candidate selector workflow
系统 MUST 提供显式 opt-in 的 DeepSense6G GPS Top8 Candidate Selector workflow。该 workflow MUST 默认覆盖 scenario31、scenario32、scenario33 和 scenario34，使用 `mapping_disabled`、`num_beams=64`、`future_beam1` mmWave power argmax label、circular beam distance，并默认使用 `support_ratio=0.15`。GPS v2 MUST 作为 frozen candidate generator，image、LiDAR、radar 或 camera AE 等 optional modalities MUST 只用于 Top8 候选内选择、重排或 miss 诊断。

#### Scenario: 默认 Top8 selector 配置
- **WHEN** 用户运行 DeepSense6G Top8 selector 默认配置
- **THEN** 系统 MUST 解析场景为 scenario31-34
- **AND** 系统 MUST 使用 64 beam circular label 语义
- **AND** 系统 MUST 将默认 support ratio 记录为 `0.15`
- **AND** 系统 MUST 将输出写入 `outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled/`

#### Scenario: 主方法只在 GPS Top8 内选择
- **WHEN** 系统训练或评估 Top8 selector 主 ablation
- **THEN** final prediction MUST 来自 GPS v2 Top8 candidate beams
- **AND** final score MUST 默认使用 `log p_gps(candidate_i) + lambda * modality_score(candidate_i)`
- **AND** image、LiDAR、radar 或 camera AE MUST NOT 作为主方法直接输出 64 类 beam logits
- **AND** residual delta correction MUST NOT 作为本 workflow 的主方法

#### Scenario: target query label 仅用于最终评价
- **WHEN** 系统构建训练集、拟合 normalization、执行 early stopping 或选择 checkpoint
- **THEN** target query label MUST NOT 被使用
- **AND** target query label MUST 只用于最终 metrics、predictions、figures 和 comparison report

### Requirement: Top8 candidate manifest
系统 MUST 提供 GPS v2 Top8 candidate manifest builder，从 GPS v2 logits 重新计算 Top8 candidates，并将 GPS context、candidate metadata、Top8 hit/miss label、nearest/oracle candidate、optional modality availability 和对齐诊断写为每样本一行。默认输出 MUST 为 `outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled/manifest/top8_candidate_manifest.csv`。

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

### Requirement: TopK candidate dataset
系统 MUST 提供 TopK candidate dataset，用于读取 Top8 candidate manifest 并返回稳定张量字段。dataset MUST 支持没有 optional modality 时运行 GPS context-only selector，并 MUST 只用 train/source/support 样本拟合 E/N/log_range/speed 等 normalization 参数。

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

### Requirement: TopKCandidateSelector model
系统 MUST 提供 `TopKCandidateSelector` 模型，消费 candidate features、GPS context、candidate log probabilities 和 optional modality features，并输出 Top8 candidate scores/probs、miss logit 和 diagnostics。该模型 MUST 保留 GPS prior fusion，且 lambda MUST 受配置约束。

#### Scenario: forward 输出形状稳定
- **WHEN** batch size 为 B 且 topk 为 8
- **THEN** 模型 MUST 输出 `final_candidate_scores: [B, 8]`
- **AND** 模型 MUST 输出 `modality_candidate_scores: [B, 8]`
- **AND** 模型 MUST 输出 `candidate_probs: [B, 8]`
- **AND** 模型 MUST 输出 `miss_logit: [B, 1]`
- **AND** diagnostics MUST 包含 lambda value、enabled modality metadata 或 candidate score stats

#### Scenario: GPS prior fusion 使用 clamp 后 lambda
- **WHEN** 模型计算 final candidate score
- **THEN** final score MUST 等于 candidate log prob 加上 clamp 到 `[0, lambda_max]` 的 lambda 乘以 modality score
- **AND** `lambda_init` 默认 MUST 使 lambda 约为 `0.5`
- **AND** `lambda_max` 默认 MUST 为 `3.0`

#### Scenario: candidate_probs 为 Top8 softmax
- **WHEN** 模型输出 `candidate_probs`
- **THEN** `candidate_probs` MUST 为 `softmax(final_candidate_scores)`
- **AND** 每行概率和 MUST 接近 1
- **AND** final beam MUST 通过 `candidate_beams[argmax(candidate_probs)]` 得到

#### Scenario: sparse 64 logits 兼容 helper
- **WHEN** 下游指标需要 `[B, 64]` logits
- **THEN** 系统 MUST 提供 sparse 64 logits 构造 helper
- **AND** candidate beam 位置 MUST 填入 final candidate scores
- **AND** 非 candidate beam 位置 MUST 填入 very negative value，例如 `-1e9`

### Requirement: CandidateAttentionSelector model
系统 MUST 提供 `CandidateAttentionSelector` 作为 ablation。candidate tokens MUST 作为 query，GPS token、camera AE pseudo-token 或 image tokens MUST 作为 key/value，输出每个 candidate 的 attention score，并使用与 MLP selector 相同的 GPS prior fusion 语义。

#### Scenario: camera AE pseudo-token 可运行
- **WHEN** 输入只有 `camera_ae_feature: [B, D_img]`、`gps_context: [B, D_gps]` 和 candidate features
- **THEN** 系统 MUST 将 camera AE feature 投影为 `[B, 1, D]` pseudo image token
- **AND** 系统 MUST 将 GPS context 投影为 `[B, 1, D]` GPS token
- **AND** candidate features MUST 投影为 `[B, 8, D]` candidate query tokens

#### Scenario: attention selector 输出形状
- **WHEN** CandidateAttentionSelector 完成 forward
- **THEN** 模型 MUST 输出 `final_candidate_scores: [B, 8]`
- **AND** 模型 MUST 输出 `candidate_probs: [B, 8]`
- **AND** 模型 MUST 输出 `miss_logit: [B, 1]` 或等价 miss diagnostics

### Requirement: TopK candidate selector losses
系统 MUST 提供 `TopKCandidateSelectorLoss`，总 loss MUST 由 candidate circular soft CE、target-in-Top8 index CE、miss BCE、GPS prior anchor KL 和 entropy regularization 组成，并 MUST 支持 hard-rank sample weighting。

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

### Requirement: Training protocol and ablation matrix
系统 MUST 支持 `target_adapt_beambench_top8_selector` 协议，并 MUST 至少支持 `support_only` 与 `source_pretrain_target_finetune` 训练模式。GPS v2 adapter/logits MUST frozen，camera AE 或 image encoder 默认 MUST frozen，训练只更新 Top8 selector 或 attention selector。

#### Scenario: 默认训练模式
- **WHEN** 用户运行默认 Top8 selector experiment
- **THEN** 系统 MUST 使用 `source_pretrain_target_finetune`
- **AND** source pretrain MUST 使用 source validation 做模型选择
- **AND** target fine-tune MUST 使用 target support 内部 validation 做模型选择
- **AND** target query MUST 只用于最终 evaluation

#### Scenario: 默认 ablation 覆盖
- **WHEN** 用户运行默认 Top8 selector experiment
- **THEN** summary MUST 至少包含 `gps_top1_baseline`、`gps_top8_oracle`、`gps_candidate_prob`、`gps_context_only_selector`、`camera_ae_gps_selector`、`camera_ae_gps_selector_anchor` 和 `top8_selector_no_gps_prior_fusion`
- **AND** 配置启用 attention 时 MUST 包含 `candidate_attention_selector`

#### Scenario: camera AE 不可用时跳过 camera ablation
- **WHEN** camera AE feature 不存在或不可稳定读取
- **THEN** 系统 MUST 跳过 `camera_ae_only_selector`、`camera_ae_gps_selector` 和 `camera_ae_gps_selector_anchor`
- **AND** summary MUST 写入 `skipped_reason`
- **AND** `gps_context_only_selector` MUST 仍然运行

#### Scenario: GPS candidate prob baseline 等价检查
- **WHEN** 系统运行 `gps_candidate_prob`
- **THEN** final prediction MUST 使用 GPS candidate prob 排序
- **AND** 该 baseline MUST 与 GPS top1 baseline 在 Top1 prediction 上等价；若不等价，系统 MUST 写出 pipeline warning

### Requirement: Top8 selector evaluation artifacts
系统 MUST 写出 Top8 selector summary、predictions、selection events、candidate rank distribution 和 metadata。所有 beam error MUST 使用 circular distance，所有主结果 MUST 与 GPS v2 r15 baseline 对比。

#### Scenario: summary_overall 与 summary_by_scene
- **WHEN** Top8 selector evaluation 完成
- **THEN** `summary_overall.csv` 和 `summary_by_scene.csv` MUST 包含 protocol、support ratio、label space、topk、train mode、ablation、sample count、DBA、DBA zero ratio、mean/median circular error、exact、pm1/pm2/pm4、top1/top3/top5、P_error_lt4、GPS baseline 指标、delta vs GPS、Top8 recall、Top8 oracle 指标、selector accuracy when target in Top8、nearest candidate selection accuracy 和 miss diagnostics

#### Scenario: summary_by_top8_hit_miss
- **WHEN** Top8 selector evaluation 完成
- **THEN** 系统 MUST 写出 `summary_by_top8_hit_miss.csv`
- **AND** 该文件 MUST 分别报告 `target_in_top8=1` 与 `target_in_top8=0` 的 count、GPS 指标、selector 指标、oracle 指标、selector exact acc、nearest candidate error 和 miss probability mean

#### Scenario: predictions 字段
- **WHEN** 系统写出 `predictions.csv`
- **THEN** 每行 MUST 包含 scene、sample id、support/query role、split role、target label、GPS top1、final top1、GPS error、final error、Top8 oracle beam/error、target_in_top8、target_candidate_index、nearest_candidate_index、selected_candidate_index、selected_candidate_rank、miss label/prob、candidate beams/probs/scores JSON、ablation、train mode、image path 和 image_exists

#### Scenario: selection events 只记录变化样本
- **WHEN** final top1 与 GPS top1 不一致
- **THEN** 系统 MUST 在 `selection_events.csv` 记录 scene、sample id、target label、GPS top1、final top1、GPS error、final error、improvement、target_in_top8、selected candidate rank、target candidate index、miss probability 和 candidate beams JSON

#### Scenario: rank distribution
- **WHEN** 系统完成 manifest 或 evaluation
- **THEN** 系统 MUST 写出 `candidate_rank_distribution.csv`
- **AND** target 不在 Top8 的样本 MUST 记录为 `rank=miss`

### Requirement: Top8 selector visualization and comparison report
系统 MUST 提供 Top8 selector plotter 与 GPS v2 comparison report。plotter MUST 从 results directory 读取结果并写入 `figures/`；comparison report MUST 自动回答 Top8 selector 是否超过 GPS v2 r15、是否接近 Top8 oracle、提升来自哪些 scene、scenario32/34 是否受 Top8 recall 上限限制、camera AE 是否优于 GPS context-only、GPS prior fusion 是否更稳、miss head 是否有效。

#### Scenario: plotter 生成标准 figures
- **WHEN** 用户运行 Top8 selector plotter
- **THEN** 系统 MUST 生成每个 scene 的 ENU scatter、GPS top1 error、selector final error、improvement、target rank distribution、selected candidate rank distribution、Top8 hit/miss spatial map、before/after residual histogram、before/after signed residual、label distribution、candidate probability calibration 和 miss probability diagnostics

#### Scenario: image montage 可用即生成
- **WHEN** image path 可用
- **THEN** plotter MUST 生成 selector 成功修正、selector 改坏样本和 Top8 miss 样本 montage
- **AND** image 不可用时 MUST 跳过 montage 并记录原因

#### Scenario: comparison report 写出路径
- **WHEN** 用户运行 GPS v2 comparison CLI
- **THEN** 系统 MUST 写出 `comparison_with_gps_v2.csv`
- **AND** 系统 MUST 写出 `comparison_report.md`
- **AND** comparison 表 MUST 包含 scene、ablation、GPS v2 DBA、selector DBA、delta DBA、GPS v2 mean error、selector mean error、delta mean error、GPS v2 Top8、Top8 oracle DBA、selector accuracy when target in Top8、Top8 hit/miss count 和 miss AUC if available

### Requirement: Top8 selector validation and documentation
系统 MUST 提供针对 manifest、selector、attention selector、loss、circular distance 和 sparse 64 logits 的测试，并 MUST 更新 README 说明 DeepSense6G GPS Top8 Candidate Selector。

#### Scenario: 单元测试覆盖核心行为
- **WHEN** 开发者运行 Top8 selector 相关测试
- **THEN** 测试 MUST 覆盖 manifest 字段、candidate soft label、wrap-around circular distance、TopKCandidateSelector forward shape、candidate probability sum、sparse 64 logits、lambda 为 0 时 GPS ranking、CandidateAttentionSelector forward shape 和 synthetic toy selector

#### Scenario: GPS baseline 复现验收
- **WHEN** 默认 Top8 selector workflow 完成
- **THEN** `gps_top1_baseline` MUST 复现 GPS v2 r15 的 DBA、Top1、Top5、Top8、Top16 和 mean error 到可解释误差范围内
- **AND** `gps_top8_oracle` exact accuracy MUST 接近 manifest 的 Top8 recall

#### Scenario: README 新增章节
- **WHEN** README 更新完成
- **THEN** README MUST 包含 “DeepSense6G GPS Top8 Candidate Selector” 章节
- **AND** 章节 MUST 说明为什么从 residual correction 转为 Top8 selector、输入输出、candidate soft label、GPS prior fusion、miss head、运行流程、结果文件和 selector 有效性判断方式
