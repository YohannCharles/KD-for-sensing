# Predictive GPS-query++ 新 strict 实验结果

## 结论

本次结果来自 2026-06-18 重新训练与重新评测的 strict run，不沿用此前 k-sweep 或旧路径产物作为主结论。协议固定为 H5/G2/F1、scene32-34、future=1、seed=17、beam64、linear DBA、test split、sample_count=1088。

Claim gate 结论为 `failed`：Predictive GPS-query++ 在 canonical P0-P5 与 GPS-query advantage slice 上均低于 `Image ResNet+GPS` 和当前 `JEPA GPS-query k=4` baseline，不能升级为优势 claim。

## Provenance

- Strict manifest: `configs/diagnostics/jepa_gps_shortcut_benchmark_predictive_gps_query_plus_plus_strict.yaml`
- Training root: `outputs/analysis/predictive_gps_query_plus_plus/strict`
- Evaluation root: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval`
- Launcher logs: `outputs/analysis/predictive_gps_query_plus_plus/launcher/20260618_131305`
- Checkpoints:
  - `image_resnet_gps`: `outputs/analysis/predictive_gps_query_plus_plus/strict/image_resnet_gps/checkpoints/best.pth`
  - `jepa_gps_query_baseline`: `outputs/analysis/predictive_gps_query_plus_plus/strict/jepa_gps_query_baseline/checkpoints/best.pth`
  - `predictive_gps_query_plus_plus`: `outputs/analysis/predictive_gps_query_plus_plus/strict/predictive_gps_query_plus_plus/checkpoints/best.pth`

旧 k=1..5 sweep 只作为背景：其中存在 seed/protocol/path provenance 与本 strict run 不完全一致的风险，因此本结果表仅采用上述新训练 checkpoint 和同一 evaluator 产物。

## Training Summary

| model | group | best epoch | best val DBA | clean test DBA | clean test Top1 |
| --- | --- | ---: | ---: | ---: | ---: |
| image_resnet_gps | resnet_image_gps | 34 | 0.8880514705882299 | 0.885723039215681 | 0.46139705881928866 |
| jepa_gps_query_baseline | jepa_gps_query_pool | 24 | 0.8828431372548958 | 0.8785539215686226 | 0.4568014705840368 |
| predictive_gps_query_plus_plus | jepa_predictive_hybrid | 37 | 0.5378063725490193 | 0.5341911764705886 | 0.2398897058801481 |

三个训练均通过 `conda run -n kd_mm_beam` 启动，最终 exit code 为 0。训练覆盖同一 seed=17、history_window=5、gps_input_source_window=2、image_history_window=5、prediction_horizon=1、scene_set=[32,33,34]、distance_metric=linear、beam_label_space=beam64。

## Canonical P0-P5

主结果表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/strict_comparison_table.csv`  
逐条件表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/p0_p5_metrics.csv`  
margin 表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/p0_p5_margins.csv`

| model | P0-P5 overall DBA | P0-P5 overall Top1 | P0 clean DBA | P4 DBA |
| --- | ---: | ---: | ---: | ---: |
| image_resnet_gps | 0.8518688725490143 | 0.39506740195715323 | 0.885723039215681 | 0.8201593137254852 |
| jepa_gps_query_baseline | 0.822865604575159 | 0.3795955882318052 | 0.8785539215686226 | 0.7886029411764667 |
| predictive_gps_query_plus_plus | 0.623856209150326 | 0.2818627450954486 | 0.5341911764705886 | 0.7099264705882332 |

Predictive GPS-query++ P-suite DBA margin:

- vs Image ResNet+GPS: -0.2280126633986883
- vs JEPA GPS-query k=4: -0.19900939542483298

## GPS-query Advantage Slice

逐条件表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/advantage_metrics.csv`  
summary 表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/advantage_summary.csv`  
margin 表: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/advantage_margins.csv`  
claim gate: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/claim_gate_summary.json`

Advantage slice 覆盖 A0/A1/A2，以及 `C3_random_async`、`C4_severe_async` 与 `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing`、`D7_joint_worst_case` 的 8 个 CxD 组合，共 11 个条件。

| model | advantage overall DBA | advantage overall Top1 | A0 DBA | C4D7 DBA |
| --- | ---: | ---: | ---: | ---: |
| image_resnet_gps | 0.8269273618538276 | 0.3660594919752451 | 0.885723039215681 | 0.7778186274509764 |
| jepa_gps_query_baseline | 0.7529411764705852 | 0.3231951871628049 | 0.8785539215686226 | 0.6291666666666657 |
| predictive_gps_query_plus_plus | 0.4406082887700541 | 0.17596925133528105 | 0.5341911764705886 | 0.2722426470588239 |

Predictive GPS-query++ advantage DBA margin:

- overall vs Image ResNet+GPS: -0.38631907308377356
- overall vs JEPA GPS-query k=4: -0.31233288770053114
- worst C4D7 vs Image ResNet+GPS: -0.5055759803921525
- worst C4D7 vs JEPA GPS-query k=4: -0.35692401960784176

Claim gate:

- `primary_claim_pass=false`
- `advantage_slice_pass=false`
- `status=failed`
- `reason=Predictive GPS-query++ underperforms both Image ResNet+GPS and JEPA GPS-query k=4 on canonical P0-P5 and on the GPS-query advantage slice.`

## Diagnostics

Runtime visual manifest: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/real_eval_manifest.json`  
Diagnostics manifest: `outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/diagnostics/manifest/predictive_gps_query_visual_diagnostics_manifest.json`

Generated figures:

- `figures/target_rank_cdf.png`: generated
- `figures/branch_weight_by_condition.png`: placeholder because condition CSV did not contain branch-weight aggregates
- `figures/latent_consistency_by_condition.png`: placeholder because condition CSV did not contain latent-consistency aggregates
- `figures/attention_entropy_by_condition.png`: placeholder because condition CSV did not contain attention aggregates

当前 evaluator 能产出真实 P0-P5、advantage、margin 和 claim gate 表，但没有把 model forward 的 branch/gate/attention diagnostics 聚合进 condition CSV。因此解释性 diagnostics 目前只有 target-rank CDF 可用；branch weight、latent consistency 和 attention 仍是后续风险/改进项。数值 claim 只使用 strict metrics 和 provenance，不依赖这些解释图。

## Implementation Note

真实 advantage evaluation 暴露了一个 baseline 兼容性问题：CxD profile 会产生 `image_dropout_mask` 等 reliability metadata，而非 predictive baseline 的 `forward()` 不接受这些 kwargs。已在 `src/kd_sensing/engine/batch.py` 中过滤模型不支持的 extra kwargs，并在 `tests/test_modality_difficulty.py` 增加回归测试，保证 baseline 可以忽略不适用的 reliability metadata，同时保留它支持的 mask。

## Reproduction Commands

训练使用同一 strict manifest 中的三个 config 和以下公共 override：

```bash
conda run -n kd_mm_beam python scripts/train.py \
  <config> \
  output.run_name=. \
  output.overwrite=true \
  output.group_by_scene=false \
  output.progress.enabled=false \
  experiment.seed=17
```

真实 per-condition evaluation 使用 ignored runtime runner：

```bash
conda run -n kd_mm_beam python outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/run_real_condition_evaluations.py --suite all
conda run -n kd_mm_beam python outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/summarize_real_condition_results.py
conda run -n kd_mm_beam python -m kd_sensing.cli.predictive_gps_query_visualizations \
  --manifest outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/real_eval_manifest.json \
  --output-dir outputs/analysis/predictive_gps_query_plus_plus/strict_real_eval/diagnostics \
  --force
```

当前环境没有安装 `kd-sensing-predictive-gps-query-visualizations` console script，因此使用 module 入口生成 diagnostics。

## Verification

- `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py -q` -> 18 passed
- `conda run -n kd_mm_beam pytest tests/test_modality_difficulty.py tests/test_gps_conditioned_jepa.py tests/test_jepa_gps_shortcut_benchmark.py -q` -> 59 passed
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` -> 65 passed
- `openspec validate add-predictive-gps-query-advantage --strict` -> passed
- `openspec status --change add-predictive-gps-query-advantage` -> all artifacts complete

未运行完整 `conda run -n kd_mm_beam pytest -q`。
