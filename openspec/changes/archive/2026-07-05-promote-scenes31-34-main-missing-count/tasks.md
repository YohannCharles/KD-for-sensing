## 1. 现有实现审阅

- [x] 1.1 阅读现有 Scene31-34 quick runner、Scene31 runner common helper、fresh eval helper 和相关 summary 脚本，确认可复用入口和输出格式。
- [x] 1.2 审阅现有 Scene31-34 manifest/config family，确认 Scene31/32/33/34 pooled training、seed、epoch、sampler 和 output root 覆盖方式。

## 2. Runner 与 fresh eval 产物

- [x] 2.1 新增或扩展 `scripts/run_scenes31_34_main.sh`，支持 `proto_seed23`、`eval_proto_all` 和 `eval_with_scene`，并实现多 GPU、跳过 complete、重跑 failed、auto-eval 和状态列表。
- [x] 2.2 确保 fresh eval 输出 `predictions_by_pattern.csv`、`pattern_metrics.csv`、`apples_to_apples_metrics.csv` 和 `checkpoint_manifest.json`，并包含 scene、missing_count、missing_ratio、available/missing modalities 等字段。

## 3. Summary、曲线与论文产物

- [x] 3.1 新增 `scripts/summarize_scenes31_34_main.py`，支持读取 `--root` 和 `--old-root`，输出 per-run、method mean/std、per-scene、missing-count curve、delta、ranking 和 conclusion。
- [x] 3.2 新增 `scripts/plot_missing_count_degradation.py`，基于 summary 输出 Top1、Within@3、MAE 和 per-scene Top1 的 PNG/PDF。
- [x] 3.3 新增 `scripts/export_scenes31_34_main_paper_tables.py`，导出主表、消融表、missing-count curve 表、scene stability 表和 paper notes。
- [x] 3.4 新增 `scripts/write_scenes31_34_main_conclusion.py`，根据 summary/table/figure 产物写出最终主结论，并如实报告实际 winner。

## 4. 文档同步

- [x] 4.1 更新主线实验文档，记录 Scene31-34 是缺失模态主实验、`prototype + random subset exposure` 是冻结主方法候选、Uniform 是 ablation、reliability fusion 和 PatternFiLM 不晋升。
- [x] 4.2 更新 inventory 或实验矩阵中对应 script/output boundary 说明，确认生成产物仍写入 ignored `outputs/`。

## 5. 验证

- [x] 5.1 运行 `openspec validate promote-scenes31-34-main-missing-count --strict`。
- [x] 5.2 运行相关 Python smoke 或 focused tests，所有项目 Python 命令必须使用 `conda run -n kd_mm_beam`。
- [x] 5.3 记录无法在本轮完成的长时间训练状态、每个方法可见 n、关键输出路径和是否还需要 AMR/AMBER 多场景外部 baseline。

## 6. 2026-07-05 主实验强化

- [x] 6.1 扩展 `scripts/run_scenes31_34_main.sh`，正式支持 `core_seed23`、`core_seed45`、`core_all_missing`、`eval_core_all`、`summarize_all`、`external_lite_seed1` 和 `external_lite_seed123`，并保留旧组名 alias。
- [x] 6.2 新增 `--slots-per-gpu`，使 GPU5/6/7 可按 `--max-parallel 6 --slots-per-gpu 2` 运行，同时写出 `runner_status.json`、`failed_runs.txt` 和 `eval_failed_runs.txt`。
- [x] 6.3 扩展 Scene31-34 main config generator，生成 core methods seed1-5 与 AMR/AMBER-lite optional seed1-3 manifest rows。
- [x] 6.4 更新 missing-count plot、paper notes 和 final conclusion，使 n=3/n=5、真实 winner、0%->75% drop、MAE@75 和 per-scene stability 均从 summary 动态读取。

## 7. 2026-07-05 论文 baseline 与成本补齐

- [x] 7.1 扩展 Scene31-34 main config generator，生成 ordinary classifier natural/subset seed1-3，并保留 classifier uniform/Bernoulli optional config。
- [x] 7.2 扩展 `scripts/run_scenes31_34_main.sh`，支持 `classifier_seed123`、`eval_all_baselines` 和 `summarize_final_all`，并新增 `--classifier-root` / `--external-root`。
- [x] 7.3 扩展 summary，读取 core、old、classifier 和 external roots，输出 final method、classifier baseline、external baseline、final delta 和 final missing-count curve。
- [x] 7.4 新增 `scripts/profile_scenes31_34_methods.py`，从已有 run artifact 导出 per-run/profile summary 和 compute cost 表。
- [x] 7.5 扩展 paper table 和 final conclusion，包含 classifier baseline、AMR/AMBER-lite maskfix 状态、compute cost 和最终 all-baseline 结论。
