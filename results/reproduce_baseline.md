# BeamBench baseline 运行记录

本文件是按时间记录的 historical log。当前可引用结论和推荐口径以 `BASELINE_REPORT.md` 开头的 Current Summary、`README_REPRODUCE.md` 当前命令和 `docs/result_claims_registry.md` 为准；本文件中的旧命令、旧 target、scene31-only、dry-run、mock、upper-bound 和 historical ablation 不会自动成为当前正式结果。

## Target Correction

User-requested target row is Arnold22 BeamBench Table III:

```text
Camera=AE, Radar=none, Lidar=none, GPS=Direct, Fusion=Yes
Scene31=0.6731, Scene32=0.6173, Scene33=0.8171, Scene34=0.7313, Overall=0.7127
```

Current project substitute:

- local data available: `dataset/DeepSense6G/scenario31`, `scenario32`, `scenario33`, `scenario34`
- config added: `configs/fusion/beambench_image_ae_gps_direct.yaml`
- encoder added: `camera_ae_frozen`, loading `CameraAutoEncoder` checkpoint
- dedicated model/training script added: `src/kd_sensing/baselines/beambench/image_ae_gps.py`
- dedicated entrypoint added: `scripts/train_beambench_image_ae_gps.py`
- throughput update: frozen AE latent cache, AMP/TF32/fused AdamW and DataLoader parallelism added for RTX 3090 + multi-core CPU runs
- status: runnable project-native local training path; trains Camera AE, freezes AE encoder, then trains GPS direct fusion classifier
- not a real official reproduction until official AE/fusion weights and exact official environment are available

## 2026-06-07 local Image AE + GPS Direct dry-run

- run type：real local DeepSense6G data, dry-run only
- target row：Camera=AE, GPS=Direct, Fusion=Yes
- environment：`kd_mm_beam`
- scene：31
- train CSV：`dataset/DeepSense6G/scenario31/train_seqs_RA_GPS_LIDAR.csv`
- test CSV：`dataset/DeepSense6G/scenario31/test_seqs_RA_GPS_LIDAR.csv`
- output dir：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast`
- checkpoint：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/checkpoints/best_image_ae_gps_direct.pt`
- AE checkpoint：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/camera_ae/checkpoints/best.pt`
- feature cache：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/feature_cache/train_camera_ae_latents.pt` 和 `test_camera_ae_latents.pt`
- predictions：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/predictions.csv`
- acceleration：CUDA AMP/TF32 enabled, frozen AE latent cache active, dry-run worker forced to 0

Command：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py --config configs/fusion/beambench_image_ae_gps_direct.yaml --dry-run --output-dir outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast
```

Metrics：

| field | value |
|---|---:|
| sample_count | 4 |
| valid_label_count | 4 |
| official_top1_acc | 0.0 |
| official_top3_acc | 0.0 |
| official_top5_acc | 0.0 |
| official_top3_dba | 0.0 |
| circular_top1_acc | 0.0 |
| circular_top3_acc | 0.0 |
| circular_top5_acc | 0.0 |
| circular_top3_dba | 0.0 |

Important: this is a dry-run with 4 train / 4 test samples and 1 AE / 1 fusion epoch. It validates the local training path only and is not a paper result.

## 2026-06-08 local Image AE + GPS Direct scene31 run

- run type：real local DeepSense6G scene31 sequence split
- target row：Camera=AE, GPS=Direct, Fusion=Yes
- environment：`kd_mm_beam`, CUDA device used
- train/test samples：5049 / 1443
- output dir：`outputs/beambench_image_ae_gps_direct_scene31`
- checkpoint：`outputs/beambench_image_ae_gps_direct_scene31/checkpoints/best_image_ae_gps_direct.pt`
- AE checkpoint：`outputs/beambench_image_ae_gps_direct_scene31/camera_ae/checkpoints/best.pt`
- predictions：`outputs/beambench_image_ae_gps_direct_scene31/predictions.csv`
- acceleration：AMP/TF32 enabled, fused AdamW requested, `num_workers=12`, frozen AE latent cache active
- selected checkpoint：epoch 35 by best `official_top3_dba`
- total fusion epochs run：50, stopped after patience window without improving best DBA

Metrics：

| field | value |
|---|---:|
| sample_count | 1443 |
| valid_label_count | 1443 |
| official_top1_acc | 0.4435204435204435 |
| official_top3_acc | 0.8205128205128205 |
| official_top5_acc | 0.9196119196119196 |
| official_top3_dba | 0.8676830676830677 |
| circular_top1_acc | 0.4435204565525055 |
| circular_top3_acc | 0.8205128312110901 |
| circular_top5_acc | 0.919611930847168 |
| circular_top3_dba | 0.9194733194733195 |

Notes：

- AE checkpoint was reused from `outputs/beambench_image_ae_gps_direct_scene31/camera_ae/checkpoints/best.pt`; the recorded AE pretraining curve reached best validation MSE `0.0009905293` at epoch 20.
- Fusion improved quickly from DBA `0.6068` at epoch 1 to `0.8466` at epoch 10, then plateaued; best DBA was reached at epoch 35.
- Local scene31 DBA is not directly comparable to Arnold22 Table III because this run uses local sequence CSV splits and local training, not official pretrained weights and official challenge test packaging.

## 2026-06-08 local paper-split Image AE + GPS Direct runs

Corrected protocol from the user: train once on scenes 32-34 and evaluate scenes 31-34 with the same checkpoint. The prior per-scene scene31 result is not comparable to Table III. The commands in this historical subsection still use `--target-beam-source future`; they are historical sequence-prediction ablations and must not be used as the current Table III strict setup.

Main strict-validation command:

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii --config configs/fusion/beambench_image_ae_gps_direct.yaml --train-scenes 32 33 34 --eval-scenes 31 32 33 34 --output-root outputs/beambench_image_ae_gps_direct_tableiii/paper_split_official_gps_future_validation --selection-split validation --fusion-val-fraction 0.1 --gps-feature-mode paper_distance_angle --target-beam-source future --num-workers 12 --ae-batch-size 128 --fusion-batch-size 512 --feature-cache-batch-size 256 --override beambench_paper.ae_checkpoint_path=outputs/beambench_image_ae_gps_direct_tableiii/paper_split_paper_calibrated_validation/camera_ae/checkpoints/best.pt
```

Strict validation result:

| Scene | Local official_top3_dba | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.3786 | 0.6731 | -0.2945 |
| 32 | 0.5031 | 0.6173 | -0.1142 |
| 33 | 0.7587 | 0.8171 | -0.0584 |
| 34 | 0.8045 | 0.7313 | +0.0732 |
| weighted overall | 0.5820 | 0.7127 | -0.1307 |

Local upper-bound command. This is both historical sequence-prediction ablation and upper-bound because it uses `test_as_validation`; it is not official unseen evaluation:

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii --config configs/fusion/beambench_image_ae_gps_direct.yaml --train-scenes 32 33 34 --eval-scenes 31 32 33 34 --output-root outputs/beambench_image_ae_gps_direct_tableiii/paper_split_official_gps_future_test_as_validation --selection-split test_as_validation --gps-feature-mode paper_distance_angle --target-beam-source future --num-workers 12 --ae-batch-size 128 --fusion-batch-size 512 --feature-cache-batch-size 256 --override beambench_paper.ae_checkpoint_path=outputs/beambench_image_ae_gps_direct_tableiii/paper_split_paper_calibrated_validation/camera_ae/checkpoints/best.pt
```

Upper-bound result:

| Scene | Local official_top3_dba | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.4295 | 0.6731 | -0.2436 |
| 32 | 0.6370 | 0.6173 | +0.0197 |
| 33 | 0.7865 | 0.8171 | -0.0306 |
| 34 | 0.7731 | 0.7313 | +0.0418 |
| weighted overall | 0.6282 | 0.7127 | -0.0845 |

Additional historical ablations:

- `paper_calibrated_relative_polar` + validation: output `outputs/beambench_image_ae_gps_direct_tableiii/paper_split_paper_calibrated_validation`, weighted overall `0.5976`.
- `paper_calibrated_relative_polar` + `test_as_validation` upper-bound: output `outputs/beambench_image_ae_gps_direct_tableiii/paper_split_paper_calibrated_test_as_validation`, weighted overall `0.5760`.
- `target_beam_source=current`: output `outputs/beambench_image_ae_gps_direct_tableiii/paper_split_current_paper_calibrated_validation`, weighted overall `0.5249`; Scene31 `0.0679`. This old run used the earlier GPS/AE setup and is retained only as historical ablation, not the current substitute protocol.

Important comparability note: the local upper-bound uses test CSV for checkpoint selection and is not official unseen evaluation. All local runs still lack official pretrained weights, official NNI/pruning search, official matching cache, and official challenge test packaging.

## 2026-06-08 scene31 generalization fix

User narrowed the goal to scene31 generalization only. This is a historical scene31-only diagnostic/historical ablation, not a current full Table III substitute. We found and fixed two local/official mismatches:

- `paper_distance_angle` now uses official `arctan(x/y)` rather than `atan2(x, y)`, avoiding a `±180` discontinuity.
- scene32 now uses the official challenge calibration angle `-0.8125375604986421 + pi/2 = 0.7583`, not the earlier `-0.76` approximation.

With the old reused 128d/64px AE. Historical sequence-prediction ablation: the command below uses `--target-beam-source future` and is not current Table III strict setup:

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii --config configs/fusion/beambench_image_ae_gps_direct.yaml --train-scenes 32 33 34 --eval-scenes 31 --output-root outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_validation --selection-split validation --fusion-val-fraction 0.1 --gps-feature-mode paper_distance_angle --target-beam-source future --num-workers 12 --ae-batch-size 128 --fusion-batch-size 512 --feature-cache-batch-size 256 --override beambench_paper.ae_checkpoint_path=outputs/beambench_image_ae_gps_direct_tableiii/paper_split_validation/camera_ae/checkpoints/best.pt
```

Result: scene31 `official_top3_dba = 0.5569`.

With a retrained 512d/64px AE. Historical sequence-prediction ablation: the command below uses `--target-beam-source future` and is not current Table III strict setup:

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii --config configs/fusion/beambench_image_ae_gps_direct.yaml --train-scenes 32 33 34 --eval-scenes 31 --output-root outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_ae512_validation --selection-split validation --fusion-val-fraction 0.1 --gps-feature-mode paper_distance_angle --target-beam-source future --num-workers 12 --ae-batch-size 128 --fusion-batch-size 512 --feature-cache-batch-size 256 --override model.primary.encoders.image.checkpoint_path= --override model.primary.encoders.image.latent_dim=512 --override beambench_paper.ae_latent_dim=512
```

Result:

| Scene | Local official_top3_dba | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.6824 | 0.6731 | +0.0093 |

This is a scene31-only result; scenes 32-34 and overall were intentionally left aside per the user request.

## 2026-06-08 full scenes 31-34 after GPS + AE fixes

The user then asked to chase scenes 32-34 and overall as well. This subsection remains a historical future-target local strict-validation record. It no longer overrides the Current Summary in `BASELINE_REPORT.md`.

Historical strict-validation checkpoint:

```text
outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_ae512_validation/checkpoints/best_image_ae_gps_direct_paper_split.pt
```

Eval-only command:

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii --config configs/fusion/beambench_image_ae_gps_direct.yaml --train-scenes 32 33 34 --eval-scenes 31 32 33 34 --output-root outputs/beambench_image_ae_gps_direct_tableiii/full_gpsfix_ae512_validation_checkpoint_eval --fusion-checkpoint outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_ae512_validation/checkpoints/best_image_ae_gps_direct_paper_split.pt --num-workers 12 --fusion-batch-size 512 --feature-cache-batch-size 256 --override model.primary.encoders.image.latent_dim=512 --override beambench_paper.ae_latent_dim=512
```

Strict-validation checkpoint eval result:

| Scene | Local official_top3_dba | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.6824 | 0.6731 | +0.0093 |
| 32 | 0.7431 | 0.6173 | +0.1258 |
| 33 | 0.8371 | 0.8171 | +0.0200 |
| 34 | 0.8158 | 0.7313 | +0.0845 |
| weighted overall | 0.7594 | 0.7127 | +0.0467 |

Full retrain strict-validation run:

- output: `outputs/beambench_image_ae_gps_direct_tableiii/full_gpsfix_ae512_validation`
- result: scene31/32/33/34 = `0.6594 / 0.7879 / 0.8471 / 0.8134`, weighted overall `0.7626`

Full retrain `test_as_validation` upper-bound:

- output: `outputs/beambench_image_ae_gps_direct_tableiii/full_gpsfix_ae512_test_as_validation`
- result: scene31/32/33/34 = `0.6756 / 0.8095 / 0.8414 / 0.8296`, weighted overall `0.7745`

At the time, the eval-only strict-validation checkpoint was preferred within the old future-target local analysis because it kept checkpoint selection on local validation while exceeding all four Table III scene DBA targets. After current-target correction, it is historical ablation only. The upper-bound remains explicitly non-official because it selects best checkpoint on test CSV.

## 2026-06-07 mock smoke

- run type：`MOCK`
- 当前仓库 commit：`00b693b9b0aa42e213884bdf3ddf36a4a25c70f8`
- 官方 BeamBench commit：`8e2c29a2afc898a69b9f9f7ece039d1e48ba60e8`
- environment：`kd_mm_beam`, Python 3.11.15, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA runtime 13.0, GPU available
- dataset split：generated mock only
- modalities declared：camera, LiDAR, radar, GPS
- mock data root：`outputs/beambench_baseline/mock_dataset`
- checkpoint：`outputs/beambench_baseline/mock_smoke/checkpoints/mock_beambench_baseline.pt`
- log：`outputs/beambench_baseline/mock_smoke/train_log.json`
- logits：`outputs/beambench_baseline/mock_smoke/mock_logits.npy`

Commands：

```bash
conda run -n kd_mm_beam python scripts/train_baseline.py --mock --epochs 2 --num-beams 64 --data-root outputs/beambench_baseline/mock_dataset --csv ml_challenge_mock_multi_modal.csv --output-dir outputs/beambench_baseline/mock_smoke --device cpu
conda run -n kd_mm_beam python scripts/check_dataset.py --data-root outputs/beambench_baseline/mock_dataset --csv ml_challenge_mock_multi_modal.csv --scene MOCK --num-beams 64 --beam-shift 0 --output outputs/beambench_baseline/mock_dataset/check_report.json
```

Metrics：

| field | value |
|---|---:|
| sample_count | 12 |
| valid_label_count | 12 |
| official_top1_acc | 0.3333333333333333 |
| official_top3_acc | 0.8333333333333334 |
| official_top5_acc | 0.9166666666666666 |
| official_top3_dba | 0.638888888888889 |
| circular_top1_acc | 0.3333333432674408 |
| circular_top3_acc | 0.8333333134651184 |
| circular_top5_acc | 0.9166666865348816 |
| circular_top3_dba | 0.8333333333333334 |

Important: this is `MOCK` smoke only. These metrics must not be used as paper, official, or leaderboard results.

## 2026-06-07 official eval plan

- run type：real official evaluation plan
- status：blocked
- official root：`/tmp/beambench-official`
- data folder requested：`dataset/DeepSense6G/raw_data/test`
- CSV requested：`ml_challenge_test_multi_modal.csv`
- type list：`radar_dense_camera_ae_gps`
- seed：42
- prediction path：`/tmp/beambench-official/results/topk/fusion_adapt_radar_dense_camera_ae_gps_42.csv`

Command：

```bash
conda run -n kd_mm_beam python scripts/eval_baseline.py --official-root /tmp/beambench-official --data-root dataset/DeepSense6G/raw_data/test --csv ml_challenge_test_multi_modal.csv --type-list radar_dense_camera_ae_gps --output-dir outputs/beambench_baseline/eval
```

Blocked reasons：

- true data CSV is unavailable locally
- official `results/models/adapt_bb_camera_ae_0_42.pth` is unavailable
- official `results/models/fusion_adapt_radar_dense_camera_ae_gps_0_42.pth` is unavailable
- official repository snapshot lacks several `.py` model/config sources referenced by `challenge.py`

No real BeamBench metric is reported for this blocked run.
