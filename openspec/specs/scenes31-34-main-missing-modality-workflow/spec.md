# scenes31-34-main-missing-modality-workflow Specification

## Purpose
定义 Scene31-34 pooled multi-scene 缺失模态主实验的 local/manual workflow 契约，包括主 runner、fresh eval 输出、summary 聚合、missing-count 退化曲线、论文表格、compute profile、最终结论和 excluded method 边界；真实训练输出、图表、表格、日志和 checkpoint 继续保留在 ignored runtime artifact 路径。
## Requirements
### Requirement: Scene31-34 主实验 runner
项目 MUST 提供 `scripts/run_scenes31_34_main.sh` 作为 Scene31-34 pooled multi-scene 主实验 local/manual runner。该 runner MUST 支持 `--group core_seed23|core_seed45|core_all_missing|eval_core_all|classifier_seed123|external_lite_seed1|external_lite_seed123|eval_all_baselines|summarize_final_all`、`--root`、`--old-root`、`--classifier-root`、`--external-root`、`--scenes`、`--gpus`、`--max-parallel`、`--slots-per-gpu`、`--train-only`、`--eval-only`、`--auto-eval`、`--overwrite-eval` 和 `--overwrite-failed`。Legacy aliases `proto_seed23`、`eval_proto_all`、`eval_with_scene` 和 `summarize_all` MAY continue to route to the equivalent current groups.

#### Scenario: proto seed23 训练集合
- **WHEN** 用户运行 `bash scripts/run_scenes31_34_main.sh --group core_seed23 --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --scenes 31,32,33,34 --gpus 5,6,7 --max-parallel 6 --slots-per-gpu 2 --auto-eval`
- **THEN** runner MUST 补跑 `proto_natural` seed2/3、`proto_sampler_uniform` seed2/3、`proto_randomdrop_bernoulli_k075` seed1/2/3 和 `proto_randomdrop_subset` seed2/3
- **AND** natural、uniform 和 subset 的 seed1 MUST 默认从 old-root 只读复用，不得覆盖
- **AND** 如果 old-root 已经存在某个目标 run 且完整，runner MUST 跳过该 run

#### Scenario: runner 并发与状态
- **WHEN** runner 在多 GPU 上执行训练或 fresh eval
- **THEN** 每张 GPU 同一时刻 MUST 运行不超过 `--slots-per-gpu` 个训练或 fresh eval worker
- **AND** 总 worker 数 MUST 不超过 `--max-parallel`
- **AND** 每个 worker MUST 设置对应的 `CUDA_VISIBLE_DEVICES`
- **AND** 已 complete 的 run MUST 默认跳过，failed run MUST 只有在 `--overwrite-failed` 时重跑
- **AND** 最后 MUST 打印并写出 completed、skipped、failed、eval_completed、eval_skipped 和 eval_failed 列表
- **AND** runner MUST 写出 `runner_status.json`、`failed_runs.txt` 和 `eval_failed_runs.txt`

#### Scenario: eval groups 不重训
- **WHEN** 用户运行 `--group eval_core_all`
- **THEN** runner MUST 只执行 fresh eval 或 per-scene eval，不启动训练
- **AND** eval MUST 使用 best checkpoint，不传入 `--max-batches`
- **AND** `--overwrite-eval` MUST 允许重写已有 eval 输出

#### Scenario: core seed45 与 all-missing
- **WHEN** 用户运行 `--group core_seed45`
- **THEN** runner MUST target natural、uniform、Bernoulli randomdrop 和 random subset exposure 的 seed4/5
- **WHEN** 用户运行 `--group core_all_missing`
- **THEN** runner MUST target natural、uniform、Bernoulli randomdrop 和 random subset exposure 的 seed1-5，并通过 old-root/new-root complete 检查跳过已完成 run

#### Scenario: summarize all
- **WHEN** 用户运行 `--group summarize_all`
- **THEN** runner MUST 依次运行 summary、missing-count plot、paper table export 和 final conclusion
- **AND** 该 group MUST NOT require `--gpus`

#### Scenario: optional external lite baselines
- **WHEN** 用户运行 `--group external_lite_seed1` 或 `--group external_lite_seed123`
- **THEN** runner MUST target AMR-lite/AMBER-lite natural/uniform optional configs
- **AND** fresh eval output SHOULD use a maskfix-marked root such as `fresh_eval_maskfix_with_scene`
- **AND** external failure MUST NOT block core prototype summary、figures、tables 或 final conclusion

#### Scenario: classifier seed123 baselines
- **WHEN** 用户运行 `--group classifier_seed123 --classifier-root outputs/scenes31_34_classifier_lmdb`
- **THEN** runner MUST target `scenes31_34_classifier_natural_es40_seed1/2/3` 和 `scenes31_34_classifier_randomdrop_subset_es40_seed1/2/3`
- **AND** generated configs MUST use ordinary CE classifier settings with prototype alignment disabled
- **AND** outputs MUST write under the classifier root, not the core proto root

#### Scenario: final all-baseline summarize
- **WHEN** 用户运行 `--group summarize_final_all`
- **THEN** runner MUST execute summary、missing-count plot、compute profile、paper table export 和 final conclusion in an order where profile is available before table export
- **AND** summary/table/conclusion MUST consume `--root`、`--old-root`、`--classifier-root` 和 `--external-root`

### Requirement: Scene31-34 fresh eval 输出缺失数量数据
Scene31-34 主实验 fresh eval MUST 为每个 run 输出 `fresh_eval_with_scene/predictions_by_pattern.csv`、`fresh_eval_with_scene/pattern_metrics.csv`、`fresh_eval_with_scene/apples_to_apples_metrics.csv` 和 `fresh_eval_with_scene/checkpoint_manifest.json`。这些文件 MUST 足以重建 pattern、scene、missing_count 和 missing_ratio 维度的指标。

#### Scenario: per-sample prediction 字段
- **WHEN** fresh eval 写出 `predictions_by_pattern.csv`
- **THEN** CSV MUST 至少包含 `run_name`、`method`、`seed`、`scene`、`sample_id`、`pattern`、`target`、`pred`、`top1_correct`、`top3_correct`、`top5_correct`、`within3_correct`、`abs_error`、`missing_count`、`missing_ratio`、`available_modalities` 和 `missing_modalities`
- **AND** 对四模态实验，missing_count=0/1/2/3 MUST 分别对应 missing_ratio=0.00/0.25/0.50/0.75

#### Scenario: pattern metrics 字段
- **WHEN** fresh eval 写出 `pattern_metrics.csv`
- **THEN** CSV MUST 至少包含 `pattern`、`missing_count`、`missing_ratio`、`available_modalities`、`missing_modalities`、`top1`、`top3`、`top5`、`within3`、`mae` 和 `num_samples`
- **AND** unsupported pattern MUST warning 并跳过，不得导致整个 eval 崩溃

#### Scenario: apples-to-apples metrics 字段
- **WHEN** fresh eval 写出 `apples_to_apples_metrics.csv`
- **THEN** CSV MUST 至少包含 `status`、`run_name`、`method`、`seed`、`checkpoint_used`、`max_batches`、`full_top1`、`miss1_top1`、`miss2_top1`、`miss3_top1`、`avg_missing_top1`、`overall_mean_top1`、`avg_missing_within@3`、`avg_missing_MAE` 和 `balanced`
- **AND** `avg_missing` MUST exclude full
- **AND** `max_batches` MUST be blank, null, none or equivalent full-eval marker for official rows

### Requirement: Scene31-34 主 summary
项目 MUST 提供 `scripts/summarize_scenes31_34_main.py`，用于读取 `--root`、可选 `--old-root`、`--classifier-root` 和 `--external-root`，并输出 Scene31-34 主实验 summary。summary MUST 支持 n=3 和 n=5 的 core prototype baselines，并能合并 classifier 与 AMR/AMBER-lite external baseline。

#### Scenario: summary 输出文件
- **WHEN** 用户运行 `python scripts/summarize_scenes31_34_main.py --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --out outputs/scenes31_34_main_lmdb/summary`
- **THEN** summary MUST 输出 `per_run.csv`、`method_mean_std.csv`、`per_scene_per_run.csv`、`per_scene_method_mean_std.csv`、`mean_over_scenes.csv`、`missing_count_curve.csv`、`missing_count_curve_by_scene.csv`、`delta_vs_randomdrop_subset.csv`、`rank_by_avg_missing_top1.md`、`rank_by_scene_stability.md` 和 `scenes31_34_main_conclusion.txt`
- **AND** final all-baseline summary MUST also output `final_method_mean_std.csv`、`final_missing_count_curve.csv`、`final_external_baselines.csv`、`final_classifier_baselines.csv` 和 `final_delta_vs_proto_subset.csv`

#### Scenario: method mean std 字段
- **WHEN** summary 写出 `method_mean_std.csv`
- **THEN** 每个 method MUST 按 seed1/2/3 聚合，并包含 full、miss1、miss2、miss3、avg_missing、overall、Within@3、MAE 和 balanced 的 mean/std 字段
- **AND** method 的 `n` MUST 等于参与 official 聚合的 ok seed 数

#### Scenario: mean over scenes 字段
- **WHEN** summary 写出 `mean_over_scenes.csv`
- **THEN** 每个 method MUST 输出 avg_missing、full、miss1、miss2、miss3、Within@3、MAE 和 balanced 的 scene-mean 字段
- **AND** per-scene 聚合 MUST 保留 scene31、scene32、scene33 和 scene34 的可见性

#### Scenario: missing count curve
- **WHEN** summary 写出 `missing_count_curve.csv`
- **THEN** 每行 MUST 包含 `method`、`n`、`missing_count`、`missing_ratio`、`top1_mean`、`top1_std`、`within3_mean`、`within3_std`、`mae_mean`、`mae_std`、`num_patterns` 和 `num_samples`
- **AND** missing_count MUST 覆盖 0、1、2、3
- **AND** 缺失 bucket 没有 pattern 时 MUST 输出 NaN 并记录 warning，不得 crash

#### Scenario: by-scene missing count curve
- **WHEN** summary 写出 `missing_count_curve_by_scene.csv`
- **THEN** 每行 MUST 包含 `scene`、`method`、`seed`、`missing_count`、`missing_ratio`、`top1`、`within3`、`mae`、`num_patterns` 和 `num_samples`

#### Scenario: final method mean std 字段
- **WHEN** summary 写出 `final_method_mean_std.csv`
- **THEN** CSV MUST include `family`、`top1_drop_0_to_75_mean`、`mae_at_75_mean`、`mask_suspect_count`、`official_ranking_included` 和 `main_read`
- **AND** official ranking MUST include core proto methods、classifier baselines and AMR/AMBER-lite rows only when `mask_suspect=false`
- **AND** reliability fusion、PatternFiLM、failed/incomplete runs and mask_suspect=true external rows MUST be excluded from official ranking

### Requirement: 缺失数量退化曲线绘图
项目 MUST 提供 `scripts/plot_missing_count_degradation.py`，从 summary 目录读取 missing count curve 并输出论文草稿图。

#### Scenario: plot 输出
- **WHEN** 用户运行 `python scripts/plot_missing_count_degradation.py --summary-root outputs/scenes31_34_main_lmdb/summary --out outputs/scenes31_34_main_lmdb/figures`
- **THEN** 脚本 MUST 输出 `fig_top1_vs_missing_count.png/pdf`、`fig_within3_vs_missing_count.png/pdf`、`fig_mae_vs_missing_count.png/pdf` 和 `fig_top1_vs_missing_ratio_by_scene.png/pdf`
- **AND** x 轴 MUST 表达 missing_count=0/1/2/3 或 missing_ratio=0%/25%/50%/75%
- **AND** Top1 与 Within@3 MUST 使用百分比，MAE MUST 保持越低越好的正常方向

#### Scenario: plot 方法集合
- **WHEN** 缺失数量曲线包含主 baseline
- **THEN** 图中 MUST 至少包含 `proto_natural`、`proto_sampler_uniform`、`proto_randomdrop_bernoulli_k075` 和 `proto_randomdrop_subset`
- **AND** best method SHOULD 使用更粗线条
- **AND** 只有 n>=3 时才画 std/error bar
- **AND** n<3 的方法 MUST 在图例中标注 `n=<value>`

### Requirement: Scene31-34 论文表格导出
项目 MUST 提供 `scripts/export_scenes31_34_main_paper_tables.py`，从主 summary 和 figures root 导出论文表格草稿与 notes。

#### Scenario: paper table 输出
- **WHEN** 用户运行 `python scripts/export_scenes31_34_main_paper_tables.py --summary-root outputs/scenes31_34_main_lmdb/summary --fig-root outputs/scenes31_34_main_lmdb/figures --out outputs/paper_tables/scenes31_34_main`
- **THEN** 脚本 MUST 输出 `table_scenes31_34_main.csv`、`table_scenes31_34_main.md`、`table_scenes31_34_ablation.csv`、`table_scenes31_34_ablation.md`、`table_scenes31_34_classifier_baseline.md`、`table_scenes31_34_external_baselines.md`、`table_scenes31_34_missing_count_curve.csv`、`table_scenes31_34_scene_stability.csv`、`table_compute_cost.md` 和 `scenes31_34_main_paper_notes.txt`

#### Scenario: main table 方法和列
- **WHEN** 脚本写出 `table_scenes31_34_main.md`
- **THEN** 表格 MUST 包含 Proto natural、Proto uniform pattern exposure、Proto Bernoulli randomdrop、Proto random subset exposure、Classifier natural、Classifier random subset、AMR-lite best available 和 AMBER-lite best available
- **AND** 列 MUST 包含 Method、Family、n、Full、Miss-1、Miss-2、Miss-3、Avg-Missing、Within@3、MAE、Drop 0%->75% 和 Main read
- **AND** Uniform MUST 标记为 ablation，不得作为 final reference

#### Scenario: classifier 和 external baseline 表
- **WHEN** 脚本写出 classifier/external baseline tables
- **THEN** classifier table MUST compare classifier natural、proto natural、classifier random subset 和 proto random subset
- **AND** external table MUST include AMR-lite natural、AMBER-lite natural、AMR-lite uniform、AMBER-lite uniform and proto random subset
- **AND** AMR/AMBER rows with missing metrics MUST show `not run`; rows with mask_suspect=true MUST show `excluded`

#### Scenario: missing count table drop
- **WHEN** 脚本写出 `table_scenes31_34_missing_count_curve.csv`
- **THEN** 表格 MUST 包含 Top1@0%、Top1@25%、Top1@50%、Top1@75%、Drop 0%->75%、Within3@0%、Within3@25%、Within3@50%、Within3@75%、MAE@0%、MAE@25%、MAE@50% 和 MAE@75%
- **AND** `Drop 0%->75%` MUST 等于 Top1@0% - Top1@75%

#### Scenario: notes 冻结主方法
- **WHEN** 脚本写出 `scenes31_34_main_paper_notes.txt`
- **THEN** notes MUST 写明 Scene31-34 是 main evaluation setting、final trusted method 是 prototype + random subset exposure、Uniform 是 ablation、reliability fusion 和 PatternFiLM 不晋升、missing_count=0/1/2/3 对应 0%/25%/50%/75%

### Requirement: Scene31-34 compute profile
项目 MUST 提供 `scripts/profile_scenes31_34_methods.py`，从已有 Scene31-34 run artifacts 生成参数量、模型大小、训练时间、推理速度和额外开销表。profile MUST NOT trigger retraining and MUST tolerate unavailable fields with NaN plus notes.

#### Scenario: profile 输出
- **WHEN** 用户运行 `python scripts/profile_scenes31_34_methods.py --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --out outputs/scenes31_34_main_lmdb/profile`
- **THEN** profile MUST output `method_profile_per_run.csv` and `method_profile_summary.csv`
- **AND** it MUST write `outputs/paper_tables/scenes31_34_main/table_compute_cost.csv` and `.md`
- **AND** proto random subset exposure MUST report `none at inference; training-only exposure strategy` as extra inference cost

### Requirement: Scene31-34 最终主结论
项目 MUST 提供 `scripts/write_scenes31_34_main_conclusion.py`，用于根据主 summary、paper tables 和 figures 写出最终主结论。

#### Scenario: conclusion 内容
- **WHEN** 用户运行 `python scripts/write_scenes31_34_main_conclusion.py --summary-root outputs/scenes31_34_main_lmdb/summary --paper-table-root outputs/paper_tables/scenes31_34_main --figure-root outputs/scenes31_34_main_lmdb/figures --out outputs/scenes31_34_main_lmdb/summary/final_main_conclusion.txt`
- **THEN** 输出 MUST 包含 Scene31-34 是 main setting、`proto_randomdrop_subset_es40` 若 seed1/2/3/4/5 后仍第一则为 final trusted method、prototype-vs-classifier baseline conclusion、random subset exposure 对 Bernoulli 的 Avg-Missing/Miss3/MAE/drop 对比、AMR/AMBER maskfix status、compute cost conclusion、reliability fusion 和 PatternFiLM 不晋升、不继续模块搜索
- **AND** 如果 `proto_randomdrop_subset` 不是实际第一，conclusion MUST 如实写出实际 winner，不得硬编码
- **AND** conclusion MUST state whether AMR/AMBER-lite need seed2/3 according to the seed1 gap rule
- **AND** conclusion MUST state that random subset exposure has no extra inference-time parameters or latency relative to the same proto model unless profile data shows otherwise

### Requirement: Scene31-34 主实验排除支线
Scene31-34 主实验 workflow MUST 明确排除本轮不继续的支线，避免主线继续搜索新模块。

#### Scenario: 不继续 excluded methods
- **WHEN** runner、summary、paper table 或 conclusion 处理主实验集合
- **THEN** reliability fusion seed2/3、PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA 和 weakKD MUST 不作为主实验方法生成、训练、排名或推广
- **AND** reliability fusion seed1 MAY 作为 dashed auxiliary curve 或 quick check 背景出现，但 MUST 不作为主曲线或 final reference

#### Scenario: AMR AMBER 可选不阻塞
- **WHEN** AMR-lite 或 AMBER-lite 多场景 maskfix 配置不可用或结果缺失
- **THEN** 主 prototype summary、figures、paper tables 和 conclusion MUST 继续生成
- **AND** 输出 MUST 将 AMR/AMBER 记录为 optional next step，而不是失败条件

### Requirement: Scene31-34 final evidence checklist
Scene31-34 主缺失模态 workflow MUST 提供 final evidence checklist，用于确认论文主结论所需的 core proto n=5、ordinary classifier baseline、AMR/AMBER-lite external-lite maskfix、fresh eval、missing-count degradation curve、per-scene stability、compute profile、paper tables 和 final conclusion artifact。Checklist 未满足时，claim status MUST 保持 pending、unverified 或 not_comparable。

#### Scenario: final all summary 缺 baseline
- **WHEN** Scene31-34 final summary 缺少 classifier baseline 或 external-lite baseline
- **THEN** summary MUST 将对应 claim 标记为 pending 或 incomplete
- **AND** paper export MUST 不将其作为完整主表结论

#### Scenario: 主方法候选冻结
- **WHEN** 继续推进 Scene31-34 主实验
- **THEN** workflow MUST 将 prototype + random subset exposure 作为当前主方法候选
- **AND** Uniform、reliability fusion、PatternFiLM 和其它候选 MUST 保持 ablation、pending 或 not promoted，除非另起 OpenSpec change 改变主线

#### Scenario: mask_suspect external rows 排除 ranking
- **WHEN** AMR/AMBER-lite external-lite fresh eval row 标记 `mask_suspect=true`
- **THEN** final ranking MUST 排除该 row
- **AND** summary MUST 保留 caveat 和排除原因

