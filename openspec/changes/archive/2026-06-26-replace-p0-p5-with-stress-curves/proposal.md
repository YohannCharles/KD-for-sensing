## Why

当前 Predictive Robustness 的 P0-P5 离散条件混合了缺失、遮挡、天气和 wrong GPS 等语义，部分条件区分度弱；例如当前帧已缺失时再随机遮挡没有额外信息，导致评估表难以回答“模型到底能扛到什么程度”。需要把低区分度 P-level 收敛为少数可解释的连续 stress curve，用于测量模型在图像缺失、图像干扰和 GPS 干扰下的抗干扰上限。

## What Changes

- **BREAKING**: `predictive_jepa_robustness` canonical evaluation 不再以 P0-P5 离散条件作为主评估口径。
- 将主评估改为 clean anchor + 三条单轴 stress curve：`image_missing`、`image_noise`、`gps_noise`。
- 将 `joint_stress` 作为可选二阶段诊断，不纳入默认主指标，避免在单轴上限未摸清前再次形成混合条件。
- 删除或降级低区分度旧条件：`P1_current_frame_missing_history_available`、`P2_semantic_occlusion_history_available`、`P5_novel_weather_history_available` 不再作为默认 claim 条件；旧 P4 语义不再叠加“缺失 + 遮挡 + wrong GPS”。
- 新增抗干扰上限指标：`S@drop<=0.02`、`S@drop<=0.05`、`AUC_retention`、`collapse_s`、`weakest_axis`。
- 保留现有统一 difficulty pipeline、benchmark manifest、strict comparability、输出产物边界和本地 `outputs/` 产物策略。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `predictive-jepa-robustness`: 将主 claim/evaluation 口径从 P0-P5 离散 regional benchmark 改为 clean anchor + stress curves，并定义抗干扰上限指标。
- `jepa-gps-shortcut-benchmark`: 扩展 benchmark manifest 与输出表契约，支持 stress suite、severity sweep、retention/AUC/collapse 指标和可选 joint stress。

## Impact

- 影响 `src/kd_sensing/data/difficulty/presets.py` 中 predictive robustness 条件定义与别名策略。
- 影响 shared difficulty operator 对图像缺失、图像噪声/退化、GPS noise/wrong/delay sweep 的参数规范化和 replay metadata。
- 影响 `src/kd_sensing/diagnostics/jepa_benchmark_predictive.py`、benchmark runner/artifact/summary 里 predictive suite 的 normalization、metrics aggregation 和 claim gate。
- 影响 `configs/diagnostics/*predictive*`、相关文档账本和测试期望。
- 不新增依赖；不提交真实数据、checkpoint、cache、日志或生成图表。
