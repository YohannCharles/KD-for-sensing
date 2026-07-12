## Why

H5/P1 temporal 对比当前让 U-Mask 方法使用 Scene31-34 分层划分，却让 AMBER 和 RMBP-MM 继承 Scene31-only 基配置，导致训练、验证和最终排名不可比较。需要在重新训练基线前把统一数据划分固化到 launcher 契约中，避免后续实验再次产生同类偏差。

## What Changes

- H5/P1 launcher 为所有方法统一覆盖 Scene31-34 train、validation 和 test scenes。
- 所有方法统一使用 `stratified_80_10_10`、`stratified_by_target_beam_per_scene`、seed 42 和相同 split source/fractions。
- dry-run 回归测试验证 AMBER、RMBP-MM 与 U-Mask 生成配置的数据划分完全一致。
- 基于新配置重新训练 seed1 AMBER 和 RMBP-MM；旧 Scene31 checkpoint 不再用于公平比较。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `temporal-window-missing`: H5/P1 temporal matrix workflow 新增跨方法统一 Scene31-34 数据划分要求。

## Impact

- 修改 `scripts/launch_h5_p1_temporal_models_v1.py` 的生成配置覆盖。
- 扩展 `tests/test_h5_p1_temporal_matrix_v1.py` 的 launcher dry-run 断言。
- 新训练产物仅写入 ignored `outputs/`，不提交数据、checkpoint 或日志。
