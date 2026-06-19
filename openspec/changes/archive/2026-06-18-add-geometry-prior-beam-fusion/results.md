## Geometry-Prior Beam Fusion Results

日期：2026-06-18  
协议：H5/G2/F1，scene32-34，future=1，seed=17，beam64，linear DBA，test sample_count=1088。

## Run Status

所有 smoke 和 strict training 均已完成，产物写入 ignored 目录：

- `outputs/analysis/geometry_prior_beam_fusion/smoke/`
- `outputs/analysis/geometry_prior_beam_fusion/strict/`

执行过程中修正了两个实验配置问题：

- `geometry_prior_prior_only` 仅使用 GPS，显式改为 `single_gru` 表征核，避免继承双模态 `early_concat_gru` 后出现 `[B,T,D]` / `[B,K,T,D]` shape mismatch。
- `geometry_prior_image_only_control` 仅使用 image，显式改为 `single_gru` 表征核。

## Smoke Results

| run | Top1 | Top3 | Top5 | DBA |
| --- | ---: | ---: | ---: | ---: |
| smoke_prior_only | 0.3336 | 0.6434 | 0.7776 | 0.7677 |
| smoke_logit_fusion | 0.3750 | 0.7610 | 0.8934 | 0.8427 |
| smoke_dba_aware | 0.3631 | 0.7243 | 0.8768 | 0.8390 |
| smoke_teacher_guided | 0.3631 | 0.7243 | 0.8768 | 0.8390 |

Smoke 结论：prior-only 可训练但低于 image+GPS 级别；logit fusion clean smoke 优于 DBA-aware/teacher-guided smoke，进入 strict matrix。

## Strict Training Checkpoints

| run | best epoch | val_adba | checkpoint |
| --- | ---: | ---: | --- |
| geometry_prior_prior_only | 64 | 0.8890 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_prior_only/checkpoints/best.pth` |
| geometry_prior_image_only_control | 23 | 0.8849 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_image_only_control/checkpoints/best.pth` |
| geometry_prior_logit_fusion | 50 | 0.8904 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_logit_fusion/checkpoints/best.pth` |
| geometry_prior_dba_aware | 25 | 0.8806 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_dba_aware/checkpoints/best.pth` |
| geometry_prior_teacher_guided | 29 | 0.8841 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_teacher_guided/checkpoints/best.pth` |
| geometry_prior_mixed_curriculum | 29 | 0.8841 | `outputs/analysis/geometry_prior_beam_fusion/strict/geometry_prior_mixed_curriculum/checkpoints/best.pth` |

## Strict Clean Evaluation

Benchmark runner 使用 `delegate_evaluate=true` 对所有 checkpoint 做 test clean evaluation。Top-K/DBA 来自真实 checkpoint forward；P-suite/advantage rows 由 runner 的 deterministic degradation model 生成，因此不作为主 claim 升级依据。

| model | group | Top1 | Top3 | Top5 | DBA | clean delta vs Image ResNet+GPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| image_resnet_gps | resnet_image_gps | 0.4614 | 0.8318 | 0.9366 | 0.8857 | 0.0000 |
| jepa_gps_query_baseline | jepa_gps_query_pool | 0.4568 | 0.8134 | 0.9338 | 0.8786 | -0.0072 |
| geometry_prior_prior_only | geometry_prior_prior_only | 0.4706 | 0.8382 | 0.9393 | 0.8855 | -0.0002 |
| geometry_prior_logit_fusion | geometry_prior_fusion | 0.4697 | 0.8263 | 0.9412 | 0.8833 | -0.0025 |
| geometry_prior_dba_aware | geometry_prior_dba_aware | 0.4200 | 0.8226 | 0.9274 | 0.8788 | -0.0069 |
| geometry_prior_teacher_guided | geometry_prior_teacher_guided | 0.4366 | 0.8171 | 0.9301 | 0.8825 | -0.0032 |
| geometry_prior_mixed_curriculum | geometry_prior_mixed_curriculum | 0.4366 | 0.8171 | 0.9301 | 0.8825 | -0.0032 |

Clean regression gate 阈值为 DBA 下降不超过 0.02。所有 geometry-prior candidates 均通过 clean gate；其中 `geometry_prior_logit_fusion` 与 `geometry_prior_prior_only` 的 clean DBA 最接近 baseline。

## Diagnostics And Claim Gate

主要产物：

- `outputs/analysis/geometry_prior_beam_fusion/strict/tables/metrics_by_condition.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/tables/robustness_summary.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/predictive_gps_query_advantage_metrics.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/geometry_prior_strict_comparison.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/geometry_prior_quality.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/geometry_prior_branch_weights.csv`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/geometry_prior_claim_gate.json`
- `outputs/analysis/geometry_prior_beam_fusion/strict/results/geometry_prior_diagnostics_bundle_manifest.json`

`geometry_prior_quality.csv` 和 `geometry_prior_branch_weights.csv` 当前将缺失 runtime branch diagnostics 标记为 `unavailable`，符合本 change 对缺失字段不得伪造的要求。

专用 claim gate 结论：

- `geometry_prior_beam_fusion` claim status: `pending`
- 原因：`delegated_clean_only_perturbations_not_real_forward`
- 解释：clean/P0 是真实 checkpoint evaluation；P-suite 和 advantage rows 仍是 deterministic degradation model，不足以升级 primary claim。

## Verification

- `conda run -n kd_mm_beam pytest tests/test_geometry_prior_beam_fusion.py tests/test_jepa_gps_shortcut_benchmark.py -q` -> 22 passed
- strict benchmark rerun succeeded and wrote all result tables/JSON under `outputs/analysis/geometry_prior_beam_fusion/strict/`

## Remaining Risk

真实逐条件 P0-P5/advantage forward evaluation 仍未实现到 runner 中。本轮可作为 clean gate 和机制诊断 evidence；primary geometry-prior robustness claim 保持 `pending`。
