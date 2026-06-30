# U-MaskBeamJEPA Evaluation Matrix

该评估矩阵用于比较 U-MaskBeamJEPA 在固定缺失模态和随机缺失模态条件下的 beam prediction 表现。它只做 evaluation 和 diagnostics，不改变模型训练、encoder 或 loss。

## Fixed Patterns

默认模态顺序来自模型配置；没有配置时使用 `image, radar, lidar, gps`。mask 约定为 `1 = available`，`0 = missing`。

默认 fixed patterns 包括：

- `full`
- `missing_<modality>`
- `only_<modality>`
- `missing_<modality_a>_<modality_b>`
- `non_gps_only`，其 mask 与 `missing_gps` 相同但 pattern name 独立保留

工具不会生成 all-missing mask。

## Random Missing

`random_missing` 接收缺失概率列表，例如 `0.25, 0.5, 0.75`。每个 batch 都会重新采样 mask，并保证每个样本至少一个模态可用。显式 pattern 名可写作 `random_0.25`、`random_0.5`、`random_0.75`。

## Output Fields

- `pattern`: pattern 名称。
- `mask`: fixed pattern 的逗号分隔 mask；random pattern 记录缺失率。
- `num_samples`: 聚合样本数。
- `loss`: hard-label cross entropy。
- `top1`, `top5`: beam Top-1 / Top-5 accuracy。
- `mean_confidence`: softmax 最大概率均值。
- `mean_global_reliability`: global reliability 均值。
- `mean_global_reliability_correct`, `mean_global_reliability_wrong`: top1 正确/错误样本的 global reliability 均值。
- `mean_modality_reliability`: modality reliability 均值。
- `mean_available_modality_reliability`: 仅统计可用模态的 modality reliability 均值。
- `ece`: 简单 expected calibration error。

## CLI

```bash
conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix \
  --config configs/fusion/u_mask_beam_jepa_s32.yaml \
  --checkpoint <checkpoint_path> \
  --output-dir outputs/eval/u_mask_beam_jepa_s32_matrix \
  --split val \
  --random-missing 0.25 0.5 0.75 \
  --prediction-index last
```

也可以使用 eval 配置：

```bash
conda run -n kd_mm_beam kd-sensing-eval-u-mask-matrix \
  --config configs/eval/u_mask_beam_jepa_s32_eval_matrix.yaml \
  --checkpoint <checkpoint_path> \
  --output-dir outputs/eval/u_mask_beam_jepa_s32_matrix
```

CLI 会保存：

- `eval_matrix.csv`
- `eval_matrix.json`
- `eval_matrix.md`

## Interpretation

`top1` 和 `top5` 越高越好。`full` 与 `missing_xxx` 的差距反映该模态对模型的贡献，`only_xxx` 反映单模态能力。理想情况下，`mean_global_reliability_correct` 应高于 `mean_global_reliability_wrong`。`mean_available_modality_reliability` 可用于观察模型对可用模态的信任程度。

这不是 corrupted modality protocol；图像噪声、GPS 偏移、异步或其他 corruption evaluation 需要单独 change。reliability diagnostics 只是评估统计，不代表已经完成校准训练。
