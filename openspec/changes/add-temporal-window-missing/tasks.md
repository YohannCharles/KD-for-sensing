## 1. 配置与 CLI

- [x] 1.1 新增 `history_window` / `prediction_window` alias 规范化，并同步 dataset/model 旧字段。
- [x] 1.2 训练、评估和 U-Mask eval matrix CLI 支持 temporal missing 参数。
- [x] 1.3 runtime metadata 和 final config 明确记录窗口与 temporal missing 配置。

## 2. Temporal Missing 实现

- [x] 2.1 新增 temporal missing difficulty operator 和注册入口。
- [x] 2.2 生成并组合 `temporal_mask`、`modality_temporal_mask`、`available_modalities`、模态 valid/dropout mask。
- [x] 2.3 实现 frame、modality-frame、block 模式和全缺失保底统计。

## 3. 脚本与文档边界

- [x] 3.1 新增 `scripts/check_temporal_window_missing.py`。
- [x] 3.2 新增 `scripts/launch_temporal_missing_v1.py` 和 `scripts/summarize_temporal_missing_v1.py`。
- [x] 3.3 更新脚本 inventory / 架构 allowlist，确保新增脚本生命周期可审计。

## 4. 验证

- [x] 4.1 新增 `tests/test_temporal_window_missing.py` 覆盖窗口 alias、mask、保底、batch shape 和 masked mean helper。
- [x] 4.2 运行 `openspec validate add-temporal-window-missing --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_temporal_window_missing.py -q`。
- [x] 4.4 运行数据检查和 launcher dry-run。
- [ ] 4.5 运行训练 smoke 命令；当前环境中 `u_mask_beam_jepa_smoke` 因既有 encoder 配置缺口失败，synthetic trainer smoke 超过 120 秒无输出后中止。

## 5. H5/P1 temporal matrix v1

- [x] 5.1 实现在线 `stratified_modality_temporal` sampler、固定 eval mask cache 和 batch 应用 helper。
- [x] 5.2 新增 `scripts/launch_h5_p1_temporal_models_v1.py`、`scripts/eval_h5_p1_temporal_matrix_v1.py` 和 `scripts/summarize_h5_p1_temporal_matrix_v1.py`。
- [x] 5.3 新增 `tests/test_h5_p1_temporal_matrix_v1.py` 覆盖 sampler、cache、launcher dry-run、matrix 输出和 summary。
- [x] 5.4 运行 `openspec validate add-temporal-window-missing --strict` 与 focused pytest。

## 6. S1-S4 temporal router matrix v1

- [ ] 6.1 在 `u_mask_beam_jepa` 中新增 `temporal_router_type`、`temporal_router_distill_weight`、`temporal_aggregation` 和 S1-S4 forward/diagnostics。
- [ ] 6.2 新增 hard oracle target helper，覆盖 S1 modality、S2 per-time modality、S3 temporal、S4 global cell，tie deterministic。
- [ ] 6.3 新增 `scripts/launch_temporal_router_s1_s4_v1.py`，默认 6 方法、3 seeds、独立 logs、manifest、failed jobs。
- [ ] 6.4 新增 `scripts/eval_temporal_router_s1_s4_matrix_v1.py`，复用固定 eval mask cache 并输出 Top1/Within@3/MAE/pattern/router/mask stats。
- [ ] 6.5 新增 `scripts/summarize_temporal_router_s1_s4_v1.py`，输出 6 方法三类矩阵、总表、router diagnostics 和自动分析。
- [ ] 6.6 新增 `tests/test_temporal_router_s1_s4_v1.py` 覆盖 sampler、S1-S4 forward、oracle、cache、eval matrix、launcher dry-run 和 summary parser。
- [ ] 6.7 运行 OpenSpec validate、launcher dry-run、focused pytest 和可行 smoke。
