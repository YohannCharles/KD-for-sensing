## Why

当前主线已支持多模态序列输入和缺失模态评估，但实验协议没有把 `history_window=5`、`prediction_window=1` 作为一等配置记录，也缺少可复现的时序缺失注入。为后续 temporal missing 鲁棒性实验，需要把窗口、mask 和诊断脚本纳入当前数据/训练/评估路径，并让新实验默认使用时序缺失配置。

## What Changes

- 新增显式 `history_window` / `prediction_window` 配置与 CLI alias，并同步到 dataset `seq_len`、model `seq_length`、dataset/model `num_pred` 和 runtime metadata。
- 新增 temporal missing difficulty operator，支持 `none`、`frame_bernoulli`、`modality_frame_bernoulli`、`block`；新实验默认使用 `modality_frame_bernoulli`、`prob=0.2`。
- 在 batch 中输出 `temporal_mask`、`modality_temporal_mask`、`available_modalities`、`history_indices`、`target_index` 等可审计字段，并将不可用输入位置置零。
- 复用现有 missing modality / random subset / pattern-balanced pipeline，使模态缺失与时序缺失组合时保留最终 mask。
- 新增本地检查、launcher 和 summary 脚本，用于 temporal missing v1 dry-run、smoke、H5/P1 temporal matrix 和 S1-S4 temporal router matrix 汇总，不触碰既有 outputs。
- 新增显式 opt-in 的 S1-S4 temporal-router 对照：S1 TemporalAgg→Modality Router、S2 Per-time Modality Router、S3 Two-level Router、S4 Global Modality-Time Router，并与 AMBER Full、RMBP-MM 在同一固定 eval mask cache 下比较。
- 新增 focused tests 覆盖窗口 alias、mask 语义、保底逻辑、batch shape 和配置入口。

## Capabilities

### New Capabilities

- `temporal-window-missing`: 显式历史/预测窗口配置、时序缺失 mask contract、训练/评估注入和 temporal missing v1 本地脚本。

### Modified Capabilities

无。

## Impact

- 影响代码：`src/kd_sensing/config/`、`src/kd_sensing/cli/`、`src/kd_sensing/data/difficulty/`、`src/kd_sensing/engine/`、`src/kd_sensing/models/u_mask_beam_jepa.py`、`scripts/` 和 focused tests。
- API/CLI：新增可选参数；默认窗口为 5/1，默认 temporal missing 为 `modality_frame_bernoulli`/`0.2`，仍可用 `temporal_missing_mode=none` 显式关闭。
- 产物：新增脚本默认写入 ignored `outputs/temporal_missing_v1/` 或 stdout，不提交 dataset、checkpoint、cache、logs 或训练输出。
