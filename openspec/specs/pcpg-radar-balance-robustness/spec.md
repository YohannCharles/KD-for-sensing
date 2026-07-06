# pcpg-radar-balance-robustness Specification

## Purpose
定义 PCPG（Pattern-Conditioned Prototype Gate）和 radar-protected branch-balanced 训练的 opt-in 缺失模态鲁棒实验边界，覆盖 hard subset weighting、checkpoint selection、oracle upper-bound 与本地 launcher/summary 产物边界，避免其成为默认训练语义或 package CLI。
## Requirements
### Requirement: PCPG opt-in fusion
系统 MUST 提供显式 opt-in 的 Pattern-Conditioned Prototype Gate（PCPG）融合能力，用于缺失模态鲁棒实验。PCPG MUST 根据 available mask、missing pattern、每模态 logits/prototype score 的可靠性统计和可扩展 per-modality features 生成 gate；不可用模态权重 MUST 为 0，可用模态权重 MUST 归一化且不得产生 NaN。默认训练和评估 MUST 不启用 PCPG。

#### Scenario: mask 约束正确
- **WHEN** 配置启用 PCPG 且 batch 中某些模态不可用
- **THEN** PCPG gate MUST 将不可用模态权重置为 0
- **AND** 单模态可用时该模态权重 MUST 为 1
- **AND** 多模态可用时可用模态权重和 MUST 为 1

#### Scenario: logits 融合可用
- **WHEN** 配置 `fusion_type: pcpg` 且 `pcpg_fuse_level: logits`
- **THEN** 模型 MUST 输出与现有 beam prediction loss 兼容的 fused logits
- **AND** diagnostics MUST 包含 gate 权重、available mask、pattern 名称或等价字段

### Requirement: Radar-protected branch-balanced training
系统 MUST 提供显式 opt-in 的 branch-balanced auxiliary supervision 和 radar-protected loss。启用后，训练 MUST 对可用 unimodal 分支计算 auxiliary CE，并允许 radar 分支使用更高权重；未启用时训练 loss 和 diagnostics MUST 与既有默认路径保持兼容。

#### Scenario: branch aux 默认关闭
- **WHEN** 配置未启用 `branch_aux_loss` 或 `radar_protect_loss`
- **THEN** 训练 MUST 不计算额外 unimodal/radar auxiliary loss
- **AND** 既有 supervised loss、extension 和 checkpoint 行为 MUST 保持不变

#### Scenario: radar aux CE 生效
- **WHEN** 配置启用 `branch_aux_loss` 和 `radar_protect_loss`
- **THEN** 训练 loss MUST 包含可用分支 auxiliary CE
- **AND** radar logits 可用时 MUST 额外计入 radar auxiliary CE
- **AND** diagnostics MUST 记录 unimodal loss mean、unimodal entropy/margin mean 和 radar auxiliary accuracy 或可解释 fallback

### Requirement: Hard subset static weighting
系统 MUST 提供显式 opt-in 的 hard subset loss weighting。第一版 MUST 至少支持 `hard_subset_weighting: static`，并为 full、image_only、lidar_only、radar_only、missing_image、miss3 和 unknown pattern 提供确定性权重；unknown pattern MUST fallback 到 1.0。

#### Scenario: static 权重排序
- **WHEN** hard subset static weighting 被启用
- **THEN** radar_only、missing_image 和 miss3 的权重 MUST 大于 full
- **AND** 未知 pattern MUST 返回 1.0

#### Scenario: 默认不重加权
- **WHEN** `hard_subset_weighting` 为 `none` 或未配置
- **THEN** 训练 MUST 不改变基础 loss 权重

### Requirement: Missing-aware checkpoint selection
系统 MUST 支持显式 checkpoint selection metric。默认 MUST 保持既有 `val_acc` 行为；当配置为 `avg_missing_top1` 或 `worst_pattern_top1` 时，系统 MUST 使用 epoch log 中对应指标选择 checkpoint，并在 checkpoint sidecar、final artifacts 或 summary 中记录实际 selection metric 和 selected epoch。

#### Scenario: avg_missing_top1 选择正确 epoch
- **WHEN** 多个 epoch log 包含 `avg_missing_top1` 且配置 `selection_metric: avg_missing_top1`
- **THEN** checkpoint selection MUST 选择该指标最高的 epoch
- **AND** sidecar MUST 记录 `selection_metric: avg_missing_top1`

#### Scenario: 默认 selection 保持不变
- **WHEN** 配置未声明 selection metric
- **THEN** 系统 MUST 保持当前 early-stopping/top1 checkpoint 行为

### Requirement: Oracle gate eval-only upper bound
系统 MUST 提供 eval-only oracle gate 模式。该模式 MUST 基于 ground-truth label 在可用 unimodal predictions/logits 中选择最接近目标 beam 的分支，并在输出中明确标注 oracle；oracle 结果 MUST 不作为真实方法默认汇总。

#### Scenario: oracle 输出标注
- **WHEN** 评估启用 `eval_oracle_gate`
- **THEN** 输出 MUST 包含 oracle full、avg_missing、image_only、lidar_only、radar_only、missing_image、miss3 或可用等价 pattern metrics
- **AND** 输出 MUST 包含 oracle chosen modality distribution
- **AND** 结果 MUST 标注为 oracle upper bound

### Requirement: PCGP radar balance experiment scripts
系统 MUST 提供本地手工实验 launcher 和 summary helper，用于 6 组 PCPG/radar-balance 实验。launcher MUST 支持 GPU 列表、总并发、每 GPU 并发、seeds、experiments、output root、dry-run、skip_completed 和 force；summary helper MUST 聚合 summary.csv、summary.md、pattern_metrics.csv 和 gate_diagnostics.csv。二者 MUST 只写 ignored runtime output root，不新增 package console script。

#### Scenario: launcher dry-run manifest
- **WHEN** 用户运行 launcher dry-run，指定 `--gpus 1,2,3,4 --max_jobs 8 --per_gpu 2`
- **THEN** launcher MUST 只打印或写出将执行的命令而不启动训练
- **AND** 分配的 GPU MUST 只来自 1、2、3、4
- **AND** 任一 GPU 的并发槽位 MUST 不超过 2，总并发 MUST 不超过 8
- **AND** job manifest MUST 包含 experiment、seed、gpu、cmd、status、start_time、end_time 和 log_path

#### Scenario: summary 输出字段
- **WHEN** summary helper 扫描 `outputs/pcpg_radar_balance_v1`
- **THEN** `summary.csv` MUST 至少包含 experiment、seed、full、avg_missing、image_only、lidar_only、radar_only、missing_image、miss1、miss2、miss3、within3、MAE、selection_metric、best_epoch、full-minus-avg_missing gap 和相对 baseline delta 字段
- **AND** `summary.md` MUST 包含 mean/std、delta、gate diagnostics 简述和目标达成判断
