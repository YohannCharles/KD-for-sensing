## 1. 上下文与现有实现审计

- [x] 1.1 阅读现有 Scene31 subset reliability runner、summary、mask diagnostics、fresh eval helper、配置生成脚本和相关 specs，确认复用点与已有 dirty tree 变更。
- [x] 1.2 检查当前 `outputs/scene31_baseline_pack_lmdb`、`outputs/scene31_subset_reliability_lmdb` 中 AMR/AMBER-lite、reliability seed1/2/3 和 PatternFiLM 产物状态，只记录状态，不删除或覆盖旧结果。

## 2. AMR/AMBER-lite maskfix formal eval 链

- [x] 2.1 补齐或新增 `scripts/run_scene31_modular_maskfix_eval.sh`，只对 complete AMR/AMBER-lite run 执行 best-checkpoint fresh eval 到 `fresh_eval_maskfix/`。
- [x] 2.2 扩展 `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`，支持 `--baseline-root`、`--gpus`、`--overwrite-eval`，并复用 maskfix eval。
- [x] 2.3 maskfix eval 后写出 `apples_to_apples_metrics.csv`、`pattern_metrics.csv`、`mask_suspect.json`、`eval_log.txt`，覆盖 maskfix_eval、mask_applied、missing_count、checkpoint provenance 和 suspect reason 字段。
- [x] 2.4 运行 `conda run -n kd_mm_beam python scripts/diagnose_modular_missing_mask.py --root outputs/scene31_baseline_pack_lmdb`，若本地数据/checkpoint 缺失则记录不可运行原因。

## 3. Scene31 combined summary 与 conclusion

- [x] 3.1 修改 `scripts/summarize_scene31_subset_reliability.py`，AMR/AMBER-lite 优先读 `fresh_eval_maskfix/`，无 maskfix 或 suspect 时设置 `excluded_from_official_ranking=true`。
- [x] 3.2 summary 输出 `maskfix_eval`、`mask_suspect`、`excluded_from_official_ranking`、`mask_suspect_reason`，并打印 AMR/AMBER-lite mask status。
- [x] 3.3 更新 promotion/conclusion 逻辑，输出 trusted reference、reliability status、PatternFiLM do-not-promote、AMR/AMBER included/excluded reason 和下一步建议。

## 4. Reliability fusion seed3 与 seed4/5 准备

- [x] 4.1 确认或生成 `proto_randomdrop_subset_reliability_fusion_es40_seed3` 配置，确保只改 seed 且禁用 condBTAPA、weakKD、MPDRO、beamsoft、PatternFiLM、AMR、AMBER。
- [x] 4.2 扩展 runner group `reliability_seed3`，支持 `--overwrite-failed`、`--auto-eval`、best checkpoint fresh eval、训练/评估日志和 CUDA illegal instruction traceback 记录。
- [x] 4.3 生成但不默认运行 seed4/5 配置，并新增 runner group `reliability_seed45`；确认默认 `all_new` 不包含 seed4/5。
- [x] 4.4 若 GPU/数据可用，运行 `bash scripts/run_scene31_subset_reliability.sh --group reliability_seed3 --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval --overwrite-failed`，否则记录未运行原因。

## 5. Scene31-34 多单场景最小验证 pipeline

- [x] 5.1 检查 Scene31、32、33、34 dataset config、scene descriptor、默认 path 和 LMDB/cache 支持，缺失项 warning。
- [x] 5.2 新增 `scripts/run_scenes31_34_subset_reliability.sh`，支持 `quick_seed1` 与 `subset_vs_reliability_seed123`，默认 root 为 `outputs/scenes31_34_subset_reliability_lmdb`。
- [x] 5.3 准备 Scene31-34 pooled quick seed1 四个 run，以及 subset/reliability seed1/2/3 显式 group；不包含 PatternFiLM、JTT、MVFR、MPDRO、beamsoft、condBTAPA、weakKD、AMR、AMBER。
- [x] 5.4 新增 `scripts/summarize_scenes31_34_subset_reliability.py`，输出 per-run、per-scene mean/std、pooled mean/std、delta、avg_missing ranking、scene stability ranking 和 conclusion。

## 6. 验证与状态记录

- [x] 6.1 运行 `openspec validate stabilize-scene31-maskfix-reliability-multiscene --strict`。
- [x] 6.2 运行相关 focused tests，所有 Python 命令使用 `conda run -n kd_mm_beam`；至少覆盖 runner/summary 静态或 fixture 检查。
- [x] 6.3 若本地资源允许，运行 maskfix eval、Scene31 summary、Scene31-34 quick seed1 和 Scene31-34 summary；无法完成长跑时在最终说明中列出原因、已生成文件和可续跑命令。

## 7. 论文级证据链整理

- [x] 7.1 扩展 Scene31-34 fresh eval，使 eval-only 重评可写 `fresh_eval_with_scene/predictions_by_pattern.csv`，且不重训、不覆盖旧 `fresh_eval/`。
- [x] 7.2 新增 `scripts/summarize_scenes31_34_per_scene.py`，从 per-sample prediction 汇总 per-scene/per-run/method/stability/delta/conclusion。
- [x] 7.3 新增 `scripts/export_scene31_paper_tables.py`，导出 Scene31 main/ablation/external baseline paper tables 与 notes。
- [x] 7.4 新增 `scripts/export_scenes31_34_paper_tables.py`，导出 Scene31-34 pooled/per-scene/stability paper tables 与 notes。
- [x] 7.5 新增 `scripts/write_final_experiment_conclusion.py`，冻结最终主线为 prototype + randomdrop subset exposure，并明确不继续 reliability/PatternFiLM/JTT/MVFR/MPDRO。
- [x] 7.6 运行 per-scene eval/summary/table/conclusion 与最小验证命令；若资源不足，记录未完成原因和续跑命令。
