# geometry-prior-beam-fusion Specification

## Purpose
Define the current opt-in GPS geometry prior and logit-fusion contract for DeepSense6G beam prediction. This capability covers label-space-aligned prior logits, assistive fusion, clean-first claim gating, teacher-guided stabilization metadata, and diagnostics bundle requirements so geometry-prior experiments can be evaluated without restoring retired residual/KD routes.
## Requirements
### Requirement: GPS geometry beam prior
系统 MUST 支持 opt-in GPS geometry beam prior 分支。该分支 MUST 从 GPS relative polar、relative Cartesian、calibrated angle、历史差分或配置声明的等价几何特征生成与 beam label space 对齐的 prior logits 或 prior distribution。

#### Scenario: 构造 beam prior logits
- **WHEN** 配置启用 `geometry_prior.enabled=true` 且 batch 提供 GPS 特征
- **THEN** 模型 MUST 输出 shape 与 beam head 对齐的 `geometry_prior_logits` 或 `geometry_prior_distribution`
- **AND** metadata MUST 记录 GPS feature mode、beam label space、calibration mode、num beams 和 prior source

#### Scenario: prior distribution 可归一化
- **WHEN** geometry prior 输出 distribution
- **THEN** 每个有效 prediction horizon 的 class 维概率和 MUST 在数值容差内等于 1
- **AND** diagnostics MUST 记录 prior entropy、top-k beams 和 prior availability mask

#### Scenario: GPS 不可用时显式降级
- **WHEN** GPS 输入缺失、mask 无效或 reliability metadata 声明 GPS 不可用
- **THEN** geometry prior branch MUST 使用配置声明的 fallback 或标记 unavailable
- **AND** 系统 MUST 不静默伪造高置信 prior

### Requirement: Geometry prior logit fusion
系统 MUST 支持在 logit 层融合 image/fusion branch 与 geometry prior branch。融合 MUST 生成最终 beam logits，并 MUST 保留每个分支的 standalone logits 用于 diagnostics。

#### Scenario: logit fusion 输出 ModelOutput 兼容
- **WHEN** geometry-prior fusion forward 成功
- **THEN** 输出 MUST 能被现有 `adapt_model_output`、loss、Top-K 和 DBA metric 消费
- **AND** 输出或 diagnostics MUST 包含 image logits、geometry prior logits、fused logits 和 branch weights 或等价 evidence

#### Scenario: prior 不能绕过 image branch
- **WHEN** 配置声明 `geometry_prior.mode=assistive` 或默认模式
- **THEN** fusion MUST 保留 image/fusion branch 对最终 logits 的贡献
- **AND** geometry prior MUST 不在默认模式下完全替代 image/fusion branch

#### Scenario: branch 维度不一致时显式投影
- **WHEN** image branch 和 geometry branch 的 horizon、class order 或 label space 不一致
- **THEN** 系统 MUST 拒绝融合或执行配置声明的显式映射
- **AND** mapping fingerprint MUST 写入 run metadata

### Requirement: Clean-first curriculum and claim gate
Geometry-prior candidate 的训练和 claim MUST 使用 clean-first curriculum。系统 MUST 在 hard-condition 或 advantage-slice claim 前检查 clean/P0 regression。

#### Scenario: clean gate 阻止 claim upgrade
- **WHEN** candidate 在 P0/clean DBA 相对 strict `Image ResNet+GPS` baseline 下降超过配置阈值
- **THEN** claim gate MUST 标记 primary claim failed 或 pending
- **AND** 系统 MUST 不允许仅凭 P-suite 子集或 advantage slice 升级 claim

#### Scenario: mixed curriculum 记录比例
- **WHEN** 训练配置启用 clean/P-suite/advantage mixed curriculum
- **THEN** final config 和 run metadata MUST 记录每类 difficulty 的采样比例、seed、condition list 和 schedule
- **AND** 诊断 MUST 能区分 clean-first warmup、P-suite mix 和 advantage mix 阶段

### Requirement: Teacher-guided stabilization
Geometry-prior training MAY 使用显式声明的 strong teacher checkpoint 做 logit/probability consistency stabilization。该行为 MUST 可关闭、可审计，并 MUST 不依赖 retired distillation runtime。

#### Scenario: teacher guidance provenance
- **WHEN** 配置启用 teacher-guided stabilization
- **THEN** metadata MUST 记录 teacher config、checkpoint path、checkpoint provenance、temperature、loss weight、detach policy 和 enabled splits
- **AND** 日志 MUST 使用 `loss/teacher_guidance`、`loss/geometry_teacher_kl` 或等价非 retired 路线命名

#### Scenario: 不恢复旧 KD 子包
- **WHEN** teacher-guided stabilization 被构建
- **THEN** 系统 MUST 不要求 `kd_sensing.distillation` 子包、旧 KD CLI、旧 KD YAML 或 retired compatibility wrapper
- **AND** 架构边界测试 MUST 能区分该 opt-in stabilization 与退役 KD research line

#### Scenario: target-side leakage boundary
- **WHEN** teacher guidance 用于 target adaptation 或 evaluation-like split
- **THEN** 配置 MUST 显式声明是否允许使用 teacher logits
- **AND** 系统 MUST 不读取 target_test label、beam power oracle 或未来信息来构造 teacher target

### Requirement: Geometry-prior diagnostics bundle
Geometry-prior strict evaluation MUST 输出 machine-readable diagnostics，用于解释 prior、image branch、teacher 和 fused prediction 的关系。

#### Scenario: 输出 prior quality 表
- **WHEN** geometry-prior evaluation 完成
- **THEN** diagnostics MUST 包含 prior standalone DBA/Top-K、prior entropy、prior-target distance 和 prior availability
- **AND** 表格 MUST 按 condition、split、seed 和 model group 分组

#### Scenario: 输出 branch agreement 表
- **WHEN** fused model 同时有 image logits、geometry prior logits 和 fused logits
- **THEN** diagnostics MUST 输出 prior-image agreement、prior-teacher agreement、fused improvement/degradation 和 branch weight summary
- **AND** 缺失字段 MUST 标记 unavailable，不得生成伪数值

#### Scenario: strict claim table 可追溯
- **WHEN** strict comparison table 生成
- **THEN** 每一行 MUST 声明 config path、weights path、sample count、metric profile、beam label space、seed、difficulty digest 和 teacher/provenance metadata
- **AND** 缺少这些字段 MUST 阻止 claim upgrade

### Requirement: Geometry-prior claim requires real perturbation forward
Geometry-prior robustness claim gate MUST distinguish real-forward evidence from delegated clean-only、synthetic metrics 或 deterministic degradation rows。只有真实 per-condition forward metrics MAY 升级 primary claim。

#### Scenario: real-forward evidence 升级 claim
- **WHEN** candidate clean gate 通过且 P-suite/advantage metrics 均来自 real-forward logits
- **THEN** geometry-prior claim gate MAY 返回 `pass`
- **AND** output MUST 记录 real-forward shard completeness、sample_count 和 cache fingerprint

#### Scenario: delegated evidence 保持 pending
- **WHEN** candidate 只有 clean delegated evaluation 或 synthetic/degradation perturbation rows
- **THEN** geometry-prior claim gate MUST 返回 `pending`
- **AND** reason MUST 包含 `delegated_clean_only_perturbations_not_real_forward` 或等价 machine-readable code

### Requirement: Geometry prior rerank diagnostics
Geometry-prior rerank evaluation MUST 输出 prior 是否进入 candidate set、是否改变最终 beam、是否改善 target rank/DBA 的 diagnostics。

#### Scenario: prior 参与候选
- **WHEN** geometry prior top-k beam 进入 rerank candidate set
- **THEN** diagnostics MUST 记录 prior candidate count、prior selected count、prior target rank 和 prior-anchor agreement
- **AND** aggregate tables MUST 可按 condition、split、seed 和 model group 分组

#### Scenario: prior 不可用
- **WHEN** GPS prior unavailable、high entropy 或 reliability metadata 声明 GPS 无效
- **THEN** reranker MUST 标记 prior branch unavailable 或 low trust
- **AND** diagnostics MUST 记录 fallback reason，不能静默声明 prior 帮助

### Requirement: Clean no-regret summary
Geometry-prior strict diagnostics MUST 输出 clean no-regret summary，量化 reranker 对 anchor clean prediction 的改变和损害。

#### Scenario: clean regression gate
- **WHEN** clean/P0 real-forward evaluation 完成
- **THEN** diagnostics MUST 输出 anchor DBA、rerank DBA、delta、changed-top1 rate 和 harmful-change rate
- **AND** clean regression 超阈值时 claim gate MUST 标记 failed 或 pending

