## 1. 配置与 CLI

- [x] 1.1 新增 `history_window` / `prediction_window` alias 规范化，并同步 dataset/model 旧字段。
- [x] 1.2 训练、评估和 U-Mask eval matrix CLI 支持 temporal missing 参数。
- [x] 1.3 runtime metadata 和 final config 明确记录窗口与 temporal missing 配置。

## 2. Temporal Missing 实现

- [x] 2.1 新增 temporal missing difficulty operator 和注册入口。
- [x] 2.2 生成并组合 `temporal_mask`、`modality_temporal_mask`、`available_modalities`、模态 valid/dropout mask。
- [x] 2.3 实现 frame、modality-frame、block 模式和全缺失保底统计。

## 3. 文档边界

- [x] 3.1 更新 temporal 配置、H5/P1 scripts inventory 和架构 guard，确保 local/manual lifecycle 可审计。

## 4. 验证

- [x] 4.1 新增 `tests/test_temporal_window_missing.py` 覆盖窗口 alias、mask、保底、batch shape 和 masked mean helper。
- [x] 4.2 运行 `openspec validate add-temporal-window-missing --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_temporal_window_missing.py -q`。
- [x] 4.4 运行 H5/P1 launcher dry-run 和 synthetic focused checks。

## 5. H5/P1 temporal matrix v1

- [x] 5.1 实现在线 `stratified_modality_temporal` sampler、固定 eval mask cache 和 batch 应用 helper。
- [x] 5.2 新增 `scripts/launch_h5_p1_temporal_models_v1.py`、`scripts/eval_h5_p1_temporal_matrix_v1.py` 和 `scripts/summarize_h5_p1_temporal_matrix_v1.py`。
- [x] 5.3 新增 `tests/test_h5_p1_temporal_matrix_v1.py` 覆盖 sampler、cache、launcher dry-run、matrix 输出和 summary。
- [x] 5.4 运行 `openspec validate add-temporal-window-missing --strict` 与 focused pytest。
