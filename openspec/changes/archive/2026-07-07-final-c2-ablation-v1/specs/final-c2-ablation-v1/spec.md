## ADDED Requirements

### Requirement: final c2 消融矩阵
系统 MUST 提供本地手工 `final_c2_ablation_v1` 实验矩阵，覆盖主方法、最强非 router 对照、negative/trade-off、router、prototype、fusion baseline 和 pattern weighting 消融。矩阵默认 MUST 生成 67 个 job：A 组 19 个、B 组 15 个、C 组 12 个、D 组 12 个、E 组 9 个。

#### Scenario: dry-run 生成完整矩阵
- **WHEN** 用户运行 `scripts/launch_final_c2_ablation_v1.py --dry_run --gpus 0,1,2,3,4,5,6,7 --max_jobs 8 --per_gpu 1`
- **THEN** launcher MUST 写出 `job_manifest.csv`
- **AND** manifest MUST 包含 67 个 job
- **AND** 所有 job 的 GPU MUST 来自 0-7
- **AND** 任一并发 wave 中每张 GPU MUST 不超过 1 个 job、总并发 MUST 不超过 8

### Requirement: 显式 opt-in ablation flags
系统 MUST 支持显式配置 router feature、prototype/head、fusion baseline 和 hard subset weighting ablation。未显式启用时默认行为 MUST 保持不变。

#### Scenario: router feature 可关闭
- **WHEN** 配置 `fusion_type: supervised_router` 且 `router_use_pattern_features=false`、`router_use_reliability_features=false` 或 `router_use_prototype_margin=false`
- **THEN** router forward MUST 不使用被关闭的特征
- **AND** forward 输出 MUST 无 NaN
- **AND** diagnostics/metadata MUST 记录对应开关状态

#### Scenario: prototype head 可替换为 classifier
- **WHEN** 配置 `head_type: classifier`
- **THEN** 模型 MUST 不用 prototype scores 作为决策 logits
- **AND** prototype alignment 和 modality prototype loss 可被关闭
- **AND** router prototype margin MUST 自动 fallback 并记录非 silent diagnostics

#### Scenario: average fusion mask 正确
- **WHEN** 配置 `fusion_type: average` 且部分模态不可用
- **THEN** fused logits MUST 等于可用模态 logits 的均值
- **AND** 单模态可用时 MUST 等于该模态 logits
- **AND** 输出 MUST 无 NaN

### Requirement: final launcher 运行边界
系统 MUST 提供 `scripts/launch_final_c2_ablation_v1.py`。launcher MUST 支持 `--dry_run`、`--skip_completed`、`--force`、`--experiments`、`--main_seeds`、`--ablation_seeds`、`--negative_seeds`、`--max_epochs`、`--output_root` 和 `--baseline_roots`，并通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 控制 GPU。

#### Scenario: 失败 job 不杀死其它 job
- **WHEN** 单个 job 返回非零
- **THEN** launcher MUST 继续等待其它已启动 job 完成
- **AND** 所有 job 完成后 MUST 写出 `failed_jobs.csv`
- **AND** launcher MUST 返回非零 exit code

### Requirement: final summary artifacts
系统 MUST 提供 `scripts/summarize_final_c2_ablation_v1.py`。summary MUST 读取 final root 和 baseline roots，输出 `summary.csv`、`summary.md`、`main_results.csv`、router/prototype/fusion/pattern/negative 消融表、`router_diagnostics.csv` 和 `pattern_metrics.csv`。

#### Scenario: summary 合并 baseline roots
- **WHEN** final root 和 baseline roots 中存在 missing-pattern metrics
- **THEN** summary MUST 聚合 seed mean/std
- **AND** MUST 计算相对 `a0_c2_full_main` 的 delta 字段
- **AND** MUST 生成包含主结果、router、prototype、fusion、pattern weighting、negative/trade-off 和最终推荐结论的 markdown

### Requirement: final focused tests
系统 MUST 提供 `tests/test_final_c2_ablation_v1.py`，覆盖 router feature ablation、prototype ablation、fusion baseline、soft_static weighting、launcher dry-run 和 summary parser。

#### Scenario: focused pytest 可运行
- **WHEN** 运行 `conda run -n kd_mm_beam pytest -q tests/test_final_c2_ablation_v1.py`
- **THEN** 测试 MUST 在不读取真实 dataset、不加载 checkpoint、不启动长训的情况下验证 final c2 ablation owner 行为
