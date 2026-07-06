# bprr-reliability-router Specification

## Purpose
TBD - created by archiving change add-bprr-reliability-router. Update Purpose after archive.
## Requirements
### Requirement: Raw confidence gate opt-in baseline
系统 MUST 提供显式 opt-in 的 `raw_conf_gate` 融合 baseline，用于评估普通 confidence gate 是否偏向强模态。该 gate MUST 只基于可用模态的 logits-derived confidence 特征和 available mask 生成权重，默认训练和评估 MUST 不启用。

#### Scenario: raw gate mask 约束
- **WHEN** 配置启用 `fusion_type: raw_conf_gate` 且 batch 中部分模态不可用
- **THEN** 不可用模态 gate MUST 为 0
- **AND** 单模态可用时该模态 gate MUST 为 1
- **AND** 多模态可用时可用模态 gate 权重和 MUST 为 1 且不得产生 NaN

#### Scenario: raw gate diagnostics
- **WHEN** raw confidence gate 完成训练或评估 forward
- **THEN** diagnostics MUST 包含各模态平均 gate、available mask 或 pattern 字段、entropy/margin 或等价 confidence 统计
- **AND** summary MUST 能报告 radar gate 是否明显低于 image/lidar

### Requirement: BPRR opt-in fusion
系统 MUST 提供显式 opt-in 的 BPRR（Beam-Prototype Reliability Router）融合能力。BPRR MUST 从 per-modality reliability features 和全局 missing-pattern features 生成 gate logits，叠加 pattern bias 后对 available modalities 做 masked softmax，并用 gate 在 logits 层融合 unimodal branch 输出。

#### Scenario: BPRR logits 融合可用
- **WHEN** 配置启用 `fusion_type: bprr` 且 `bprr_fuse_level: logits`
- **THEN** 模型 MUST 输出与现有 beam prediction loss 兼容的 fused logits
- **AND** diagnostics MUST 包含 BPRR gate、available mask、pattern id 或 pattern name、reliability feature summary

#### Scenario: BPRR mask 约束
- **WHEN** BPRR 收到 available mask
- **THEN** 不可用模态 gate MUST 为 0
- **AND** 单模态可用时该模态 gate MUST 为 1
- **AND** 多模态可用时可用模态 gate 权重和 MUST 为 1 且不得产生 NaN

#### Scenario: prototype feature fallback
- **WHEN** 当前模型无法提供稳定的 per-modality prototype distance 或 prototype margin
- **THEN** BPRR MUST 使用 logits-derived reliability feature fallback 继续运行
- **AND** 代码或 diagnostics MUST 清楚标注 prototype distance / margin 暂未接入

### Requirement: BPRR calibration
系统 MUST 支持显式 opt-in 的 BPRR calibration。`bprr_calibration: temperature` MUST 为每个模态维护独立正温度；默认 `none` MUST 保持既有 logits 和 gate 行为。

#### Scenario: temperature 为正且按模态独立
- **WHEN** BPRR temperature calibration 被启用
- **THEN** 每个模态 MUST 拥有独立 temperature 参数
- **AND** temperature MUST 通过 softplus 或等价机制保持为正
- **AND** forward MUST 不产生 NaN

#### Scenario: calibration diagnostics 可保存
- **WHEN** BPRR 运行结束或 summary 聚合 diagnostics
- **THEN** 系统 MUST 能写出 `modality_temperatures.json` 或等价 diagnostics 字段
- **AND** 字段 MUST 包含 image、lidar、radar、gps 或当前启用模态的温度值

### Requirement: BPRR gate regularization
系统 MUST 支持显式 opt-in 的 BPRR gate balance regularization 和 radar gate floor regularization。两个正则默认权重 MUST 为 0，只在训练时启用；评估时 MUST 不强制 gate floor。

#### Scenario: gate balance 默认关闭
- **WHEN** `bprr_gate_balance_weight` 为 0 或未配置
- **THEN** 训练 MUST 不增加 gate balance loss
- **AND** 既有训练 loss 行为 MUST 保持兼容

#### Scenario: radar gate floor 正则
- **WHEN** radar 可用、pattern 属于配置的 hard patterns 且 radar gate 低于 `bprr_radar_gate_floor`
- **THEN** radar gate regularization loss MUST 大于 0
- **AND** 当 radar 不可用、radar_only 或 radar gate 高于 floor 时，该正则 MUST 为 0 或不启用

### Requirement: Oracle gate eval-only upper bound
系统 MUST 提供 eval-only oracle gate 模式，用 ground-truth beam label 在可用 unimodal branch 中选择预测最接近目标的分支。oracle 输出 MUST 明确标注为 upper bound，且 MUST 不作为真实方法默认排名。

#### Scenario: oracle 选择可用最接近分支
- **WHEN** 评估启用 `eval_oracle_gate`
- **THEN** oracle MUST 只在可用模态中选择 unimodal branch
- **AND** 被选择分支的预测 MUST 是与 ground-truth beam 距离最小的可用分支

#### Scenario: oracle 输出完整指标和分布
- **WHEN** oracle eval 写出结果
- **THEN** 输出 MUST 至少包含 full、drop-1、drop-2、drop-3、drop1-3、avg_missing、image_only、lidar_only、radar_only、missing_image 或等价 pattern metrics
- **AND** 输出 MUST 包含 oracle chosen modality distribution
- **AND** 输出 MUST 标注 `oracle`

### Requirement: BPRR reliability-router experiment scripts
系统 MUST 提供本地手工实验 launcher 和 summary helper，用于 `e3/e7/e8/e9/e10/e11/e12` 七组实验。launcher MUST 支持 GPU 列表、总并发、每 GPU 并发、seeds、experiments、output root、dry-run、skip_completed、force、max_epochs override、job log 和 manifest；summary helper MUST 聚合 BPRR summary、gate diagnostics、oracle summary 和 baseline delta。

#### Scenario: launcher dry-run manifest
- **WHEN** 用户运行 BPRR launcher dry-run，指定 `--gpus 0,1,2,3,4,5,6,7 --max_jobs 8 --per_gpu 1`
- **THEN** launcher MUST 不启动训练或评估进程
- **AND** 分配的 GPU MUST 只来自 0-7
- **AND** 任一 GPU 的并发槽位 MUST 不超过 1，总并发 MUST 不超过 8
- **AND** job manifest MUST 包含 experiment、seed、gpu、cmd、status、start_time、end_time、return_code 和 log_path

#### Scenario: summary 输出字段
- **WHEN** summary helper 扫描 `outputs/bprr_reliability_router_v1`
- **THEN** `summary.csv` MUST 至少包含 experiment、seed、full、drop1、drop2、drop3、drop1_3_mean、avg_missing、image_only、lidar_only、radar_only、missing_image、within3、MAE、selection_metric、best_epoch、gate_entropy、mean_gate_image、mean_gate_lidar、mean_gate_radar 和 radar gate pattern fields
- **AND** `summary.md` MUST 包含相对 e5 baseline、相对 e6 robustness-first、raw gate vs BPRR、BPRR vs oracle、hard subset / JEPA ablation 和 gate collapse 判断

