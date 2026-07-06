## 1. 代码定位与边界确认

- [x] 1.1 定位当前训练入口、U-MaskBeamJEPA fusion、weighted_sum/reliability diagnostics、prototype/head、missing mask/sampler、checkpoint selection、Scene31-34 scripts、evaluation 输出和现有 tests。
- [x] 1.2 确认新增脚本为 local/manual surface，输出边界为 ignored `outputs/pcpg_radar_balance_v1/`，不新增 package console script。

## 2. PCPG 与模型 diagnostics

- [x] 2.1 在 U-MaskBeamJEPA 中新增 `fusion_type: pcpg` 和 `pcpg_fuse_level: logits`，实现 masked softmax gate 和 logits/prototype-score 融合。
- [x] 2.2 输出 PCPG diagnostics：gate weights、available mask、pattern/gate mean、unimodal logits 或 prototype scores、entropy/margin 等可用统计。
- [x] 2.3 对 `pcpg_fuse_level: features` 给出清晰不支持或低风险 fallback，不静默声称 feature-level 已实现。

## 3. 训练目标与 checkpoint selection

- [x] 3.1 新增 branch-balanced/radar-protected training extension，支持 `branch_aux_loss`、`radar_protect_loss`、`unimodal_aux_weight`、`radar_aux_weight` 和 radar CE fallback。
- [x] 3.2 新增 hard subset static weighting helper，覆盖 full、image_only、lidar_only、radar_only、missing_image、miss3 和 unknown fallback。
- [x] 3.3 接入可选 JEPA latent alignment 或清晰 fallback，默认关闭并记录 diagnostics。
- [x] 3.4 新增 `selection_metric: val_acc|avg_missing_top1|worst_pattern_top1` 的 checkpoint selection helper，默认保持现有行为，并写入 sidecar/final summary。

## 4. Eval oracle、launcher 与 summary

- [x] 4.1 新增 eval-only oracle gate helper或脚本路径，基于 label 选择最接近 ground truth 的可用 unimodal 分支，并输出 oracle pattern metrics 与 chosen-modality distribution。
- [x] 4.2 新增 `scripts/launch_pcpg_radar_balance_v1.py`，支持 GPU/per-GPU/total 并发、seeds、experiments、dry-run、skip_completed、force、日志和 manifest。
- [x] 4.3 新增 `scripts/summarize_pcpg_radar_balance_v1.py`，输出 summary.csv、summary.md、pattern_metrics.csv 和 gate_diagnostics.csv。

## 5. 测试与验证

- [x] 5.1 新增 PCPG gate mask focused test，覆盖不可用为 0、单模态为 1、多模态和为 1、无 NaN。
- [x] 5.2 新增 hard subset weighting focused test，覆盖 hard pattern 权重大于 full 和 unknown fallback。
- [x] 5.3 新增 checkpoint selection focused test，验证 avg_missing_top1 选择正确 epoch。
- [x] 5.4 新增 launcher dry-run focused test，验证 GPU 分配、并发约束和 manifest 字段。
- [x] 5.5 运行 `openspec validate add-pcpg-radar-balance-robustness --strict`、`conda run -n kd_mm_beam pytest <focused tests> -q` 和 `conda run -n kd_mm_beam python scripts/verify_compile.py`。
