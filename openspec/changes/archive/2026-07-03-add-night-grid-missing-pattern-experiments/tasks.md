## 1. Pattern 条件 BTAPA 与 missing pattern 基础

- [x] 1.1 确认或补齐 `src/kd_sensing/utils/missing_patterns.py` 的标准 mask/name/list/classification API，并迁移需要的调用点。
- [x] 1.2 实现 pattern-conditional BTAPA sample-wise loss，记录 ordinary/btapa/active-ratio/total proto diagnostics。
- [x] 1.3 确认现有 proto 和 BTAPA tau1 三 seed 配置存在且不被新增配置覆盖。

## 2. Sampler、Reweight 与 Mask Adapter

- [x] 2.1 实现 configurable missing pattern sampler，支持 uniform、oversample 和 curriculum，并写出 epoch pattern count CSV。
- [x] 2.2 实现 hard pattern CE reweight，默认只加权 CE，并记录 weighted CE 和 avg sample weight。
- [x] 2.3 实现 lightweight mask-conditioned adapter，保持默认关闭，记录 adapter 参数量。

## 3. Night Grid 配置与 Launcher

- [x] 3.1 新增 `configs/scene31/templates/main_v3_proto_es20_base.yaml` 或复用已有 es20 base。
- [x] 3.2 新增 `scripts/generate_experiment_grid.py`，生成 A-F 58 个 run 和 6 个 baseline/reference manifest 行。
- [x] 3.3 运行生成脚本确认 64 个配置/manifest 可生成且默认不覆盖已有文件。
- [x] 3.4 新增 `scripts/run_night_grid_8gpu.sh`，支持 dry-run、过滤、skip_completed、auto_resume、stagger、eval/analysis after train 和 failed/completed 记录。

## 4. Fresh Eval、Analysis 与 Summary

- [x] 4.1 新增 `scripts/eval_night_grid.py`，复用统一 checkpoint resolver 和 missing pattern，输出 metrics/Markdown/checkpoint manifest。
- [x] 4.2 新增 `scripts/analyze_night_grid.py`，输出 by-run/by-group/mean-std/delta/top candidates/paper observations。
- [x] 4.3 增强 `scripts/summarize_missing_runs.py` 兼容 night grid manifest 字段和状态识别。

## 5. Weak KD 与 Lightweight Latent Probe

- [x] 5.1 实现 weak-pattern KD opt-in runtime，记录 kd_loss 和 kd_active_ratio，eval 不启用 teacher branch。
- [x] 5.2 实现 lightweight latent prediction probe opt-in runtime，记录 latent_pred_loss 和 latent_pred_active_ratio。

## 6. 验证

- [x] 6.1 运行 `openspec validate add-night-grid-missing-pattern-experiments --strict`。
- [x] 6.2 运行生成脚本和 night-grid launcher dry-run 验收命令。
- [x] 6.3 运行相关 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q` 或新增/相关 smoke。
- [x] 6.4 汇总未运行的长训练、真实 checkpoint eval 或 dataset 依赖项及原因。
