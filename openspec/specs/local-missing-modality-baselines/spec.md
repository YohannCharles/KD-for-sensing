# local-missing-modality-baselines Specification

## Purpose
约束 AMBER-lite、RMBP-MM 和 TII-VLRG-style 缺失模态 baseline 在本仓库实验场景中的本地训练、验证、配置、metadata 和 claim 边界。
## Requirements
### Requirement: 本地可训练 baseline 入口
系统 MUST 为 AMBER-lite、RMBP-MM 和 TII-VLRG-style baseline 提供本仓库内可训练配置。默认训练入口 MUST 使用 `kd-sensing-train --config`，且 MUST NOT 依赖官方源码、外部 checkpoint 或自动下载权重。

#### Scenario: 使用本地配置训练
- **WHEN** 用户选择任一 local missing-modality baseline 配置
- **THEN** 系统 MUST 能通过 `conda run -n kd_mm_beam kd-sensing-train --config <config>` 构建模型并进入训练/验证流程
- **AND** 默认配置 MUST 使用本地 scratch 权重或用户显式提供的本地 checkpoint

#### Scenario: external wrapper 不作为训练前置
- **WHEN** TII 或 WCL external/audit artifact 缺失
- **THEN** 本地 baseline 训练配置 MUST 仍可独立构建
- **AND** 系统 MUST NOT 将 external wrapper 缺失解释为本地 baseline 不可用

### Requirement: 缺失模态训练扰动接入
系统 MUST 使用现有 difficulty pipeline 表达训练期缺失模态扰动，不得保留训练循环不消费的伪配置字段作为唯一 dropout 声明。

#### Scenario: 训练期 dropout 生效
- **WHEN** baseline 配置声明 missing-modality train profile
- **THEN** `BatchStepRunner` 的训练 batch MUST 通过 `apply_configured_difficulty` 应用该 profile
- **AND** 输入 mask metadata MUST 能被 opt-in 模型消费或由 zero-imputation baseline 显式记录

### Requirement: baseline claim 边界
系统 MUST 将这些条目标记为 local experimental baseline。文档、metadata 和 summary MUST NOT 声称 official reproduction 或论文数值复现。

#### Scenario: 输出和文档标记为 local baseline
- **WHEN** baseline 配置、文档或 summary 描述 AMBER-lite、RMBP-MM 或 TII-VLRG-style 条目
- **THEN** 它们 MUST 标记为 local experimental baseline
- **AND** official/external/audit wrapper MUST 仅作为可选参考路径出现

### Requirement: AMBER full 纳入本地缺失模态 baseline 家族
系统 MUST 将 AMBER full architecture reproduction 作为 local missing-modality baseline 家族中的本地架构复现条目。该条目 MUST 使用当前训练入口和现有 missing/difficulty metadata 边界，不得依赖官方源码、外部 checkpoint 或专用训练循环。

#### Scenario: 使用本地训练入口启动 AMBER full
- **WHEN** 用户选择 AMBER full architecture config
- **THEN** 系统 MUST 能通过 `conda run -n kd_mm_beam kd-sensing-train --config <config>` 构建模型并进入训练/验证流程
- **AND** 默认配置 MUST 使用本地 scratch 权重或用户显式提供的本地 checkpoint

#### Scenario: AMBER full claim 保持本地实验状态
- **WHEN** 文档、summary 或 claim row 描述 AMBER full
- **THEN** 它 MUST 标记为 local architecture reproduction 或 local experimental baseline
- **AND** 缺少严格可比真实评估时 MUST NOT 声称 official AMBER reproduction

### Requirement: Local baseline stress comparability
本地缺失模态 baseline MUST 能声明 stress benchmark comparability metadata。适用对象包括 AMBER-lite、AMBER full、RMBP-MM、U-MaskBeamJEPA/weighted_sum 和其它 current local missing-modality baseline。

#### Scenario: baseline 声明 required fields
- **WHEN** baseline 被纳入 missing-modality stress suite
- **THEN** baseline manifest 或 run metadata MUST 声明 config path、weights path、checkpoint provenance、modalities、split、sample_count、label_space、metric_profile、target_source、seed 和 difficulty_digest
- **AND** 缺失 required field MUST 阻止该 baseline 进入 strict claim comparison

#### Scenario: local substitute 状态保留
- **WHEN** AMBER、RMBP-MM 或其它外部论文 baseline 使用本仓库 local implementation
- **THEN** stress summary MUST 保留 `local experimental baseline` 或 `local substitute` 状态
- **AND** 系统 MUST 不将其描述为 official reproduction

#### Scenario: baseline 缺某模态
- **WHEN** stress suite 包含 baseline 不支持的模态或 condition
- **THEN** 对应 row MUST 标记为 unavailable、not_applicable 或 not_comparable
- **AND** summary MUST 不把缺失 row 当作 0 分或 clean 结果

### Requirement: AMR-lite local baseline
系统 MUST 提供 AMR-lite local missing-modality baseline，用于表达 available modality masking、missing modality imputation 和轻量 channel/modality attention。该 baseline MUST 标记为 local experimental baseline，不得声明为完整外部论文复现。

#### Scenario: AMR-lite 构建
- **WHEN** 配置声明 `amr_lite`
- **THEN** 系统 MUST 能构建支持 image、LiDAR、radar 和 GPS modality feature 的轻量 fusion 模型或组件
- **AND** 缺失 modality MUST 使用 `zero`、`mean_feature` 或 `learnable_token` imputation 之一，默认 MUST 支持 `learnable_token`
- **AND** 模型 MUST 使用 availability/missing mask 参与 gate 或 attention 计算

#### Scenario: AMR-lite gate stats
- **WHEN** AMR-lite 完成一个训练 epoch 或可用诊断阶段
- **THEN** run 目录 MUST 写入 `amr_lite_gate_stats.csv`
- **AND** CSV MUST 包含 `epoch`、`pattern`、`modality`、`mean_gate` 和 `std_gate`

#### Scenario: AMR-lite 防泄漏
- **WHEN** 某 modality 在训练或评估 batch 中被标记为缺失
- **THEN** AMR-lite MUST NOT 使用该 modality 的真实 feature 作为可见输入
- **AND** imputation 和 gate 计算 MUST 只依赖缺失 token、可见 feature 和 mask metadata

### Requirement: AMBER-lite baseline-pack training comparison
AMBER-lite 在 baseline pack 中 MUST 支持 natural 或 randomdrop training 与 pattern-balanced exposure training 的公平对照。该对照 MUST 继续保持 AMBER-lite local-lite scope，不升级为 AMBER full 或官方复现。

#### Scenario: AMBER-lite pattern-balanced 对照
- **WHEN** baseline pack 生成 AMBER-lite run matrix
- **THEN** 矩阵 MUST 包含 `amber_lite_uniform_es40_seed1/2/3`
- **AND** 矩阵 MUST 包含 `amber_lite_natural_es40_seed1/2/3` 或 `amber_lite_randomdrop_subset_es40_seed1/2/3`

#### Scenario: AMBER-lite 参数统计
- **WHEN** AMBER-lite baseline 被 summary 读取或模型可构建
- **THEN** 系统 MUST 统计或保留 `total_params`、`trainable_params` 和 `extra_params_vs_proto`
- **AND** 参数来源 MUST 区分真实 module 统计和 unavailable

### Requirement: FeatureMod-lite local baseline
系统 MUST 支持将 FeatureMod-lite 作为可选运行组纳入 baseline pack，用于对比 missing-modality feature adaptation。FeatureMod-lite MUST 使用 missing modalities 作为 condition，不得复用失败的 PatternFiLM d8 作为同一方法换名。

#### Scenario: FeatureMod-lite 构建
- **WHEN** 配置声明 `featuremod_lite`
- **THEN** 系统 MUST 使用 missing modalities condition 生成轻量 adapter 或 affine feature correction
- **AND** 默认 adapter 维度 MUST 保持小参数量，例如 `adapter_dim=16`

#### Scenario: FeatureMod-lite baseline pack run
- **WHEN** 用户选择 baseline pack `featuremod` group
- **THEN** runner MUST 尝试运行 `featuremod_lite_uniform_es40_seed1/2/3`
- **AND** 若 FeatureMod-lite 未实现或只完成 quick screen，summary MUST 标记 skipped 或 quick_screen，不得纳入最终多 seed claim

### Requirement: Baseline pack 参数量比较
本地缺失模态 baseline 被纳入 baseline pack summary 时 MUST 提供可审计参数量字段。参数比较 MUST 支持 proto、AMR-lite、AMBER-lite 和 FeatureMod-lite。

#### Scenario: 参数字段
- **WHEN** baseline pack summary 写出 `params_comparison.csv`
- **THEN** 每个可用 method MUST 包含 `total_params`、`trainable_params` 和 `extra_params_vs_proto`
- **AND** 无法确认参数量的历史或缺失 run MUST 使用空值并记录 warning，不得伪造数值

### Requirement: Modular-lite missing-mask fresh eval diagnostics
AMR-lite and AMBER-lite local baseline fresh eval MUST verify that missing-pattern masks are received and affect outputs. Results where full and missing outputs or metrics are indistinguishable MUST be marked suspect and excluded from official winner ranking.

#### Scenario: diagnostics report mask path
- **WHEN** `scripts/diagnose_modular_missing_mask.py` is run against a baseline root
- **THEN** it MUST write `modular_missing_mask_diagnostics.csv`
- **AND** each row MUST include model name, run name, forward signature, whether missing-mask kwargs are accepted, whether eval passes a mask, whether batch filtering drops it, whether forward applies it, full-vs-missing equality and a diagnosis

#### Scenario: identical logits warn
- **WHEN** the diagnostic can compare full and missing pattern logits on the same batch
- **THEN** exactly equal full and missing logits MUST produce a warning diagnosis
- **AND** unsupported or unavailable checks MUST be explicit rather than reported as ok

#### Scenario: maskfix fresh eval does not retrain
- **WHEN** the maskfix eval runner processes AMR-lite or AMBER-lite runs
- **THEN** it MUST load the existing best checkpoint for complete runs
- **AND** it MUST NOT start training or overwrite old checkpoint files
- **AND** it MUST NOT pass `--max-batches`

#### Scenario: suspect results excluded
- **WHEN** full and missing pattern metrics remain exactly identical after maskfix fresh eval
- **THEN** the run MUST be marked `mask_suspect=true`
- **AND** summary scripts MUST exclude that run from official winner ranking and promotion decisions

### Requirement: Modular-lite formal maskfix fresh eval artifacts
AMR-lite and AMBER-lite formal maskfix evaluation MUST write a separate `fresh_eval_maskfix/` artifact set and MUST NOT overwrite existing `fresh_eval/` results or checkpoints.

#### Scenario: maskfix eval writes required files
- **WHEN** the maskfix eval runner processes a complete AMR-lite or AMBER-lite run with a best checkpoint
- **THEN** it MUST write `fresh_eval_maskfix/apples_to_apples_metrics.csv`, `fresh_eval_maskfix/pattern_metrics.csv`, `fresh_eval_maskfix/mask_suspect.json` and `fresh_eval_maskfix/eval_log.txt`
- **AND** it MUST record `maskfix_eval=true`, run name, method, checkpoint policy and checkpoint path

#### Scenario: maskfix eval skips unavailable runs
- **WHEN** an AMR-lite or AMBER-lite run directory, config or checkpoint is missing
- **THEN** the runner MUST skip that run with a warning
- **AND** it MUST NOT start training or create replacement checkpoints

#### Scenario: old eval is preserved
- **WHEN** `fresh_eval/` already exists for a modular-lite run
- **THEN** maskfix evaluation MUST write to `fresh_eval_maskfix/`
- **AND** it MUST NOT delete, mutate or reinterpret the old `fresh_eval/` directory as maskfix evidence

### Requirement: Modular-lite mask suspect artifact
AMR-lite and AMBER-lite maskfix evaluation MUST automatically mark suspicious results and expose the reason in machine-readable artifacts.

#### Scenario: identical metrics are suspect
- **WHEN** full, missing_gps, radar_only and lidar_only core metrics are exactly identical after maskfix evaluation
- **THEN** `mask_suspect.json` MUST contain `mask_suspect=true`
- **AND** the reason MUST mention identical core metrics

#### Scenario: identical logits are suspect
- **WHEN** the evaluation can compare full logits with missing-pattern logits and they are exactly equal
- **THEN** `mask_suspect.json` MUST contain `logits_full_vs_missing_equal=true`
- **AND** the run MUST be marked suspect

#### Scenario: missing mask application is required
- **WHEN** any evaluated missing pattern has `mask_applied=false`, an incorrect `missing_count`, or `maskfix_eval` is not true
- **THEN** the run MUST be marked suspect
- **AND** the suspect reason MUST be written to `mask_suspect.json`

#### Scenario: non-suspect artifact records checked patterns
- **WHEN** no suspect condition is found
- **THEN** `mask_suspect.json` MUST contain `mask_suspect=false`, an empty reason, checked patterns and identical metric groups
- **AND** the artifact MUST still record whether logits equality was checked or unavailable

