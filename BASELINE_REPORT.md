# BeamBench baseline 复现报告

## Current Summary：Arnold22 Table III 本地 substitute 口径

本节是当前推荐口径的优先入口；后续章节保留历史流水账和 ablation，不得覆盖本节。

目标行：Arnold22 BeamBench Table III `Camera=AE, Radar=none, Lidar=none, GPS=Direct, Fusion=Yes`。

当前本地 substitute 协议必须同时满足：

- target：`beam_target_source=current` 或配置中等价 `target_beam_source: current`
- window：`seq_len=1`、`num_pred=1`
- GPS Direct：`paper_distance_angle`，使用 scene paper calibration angle
- metric：linear/non-circular DBA，Top-1/3/5，字段以 `official_top3_dba` 或 `beambench_linear_topk` 对齐
- selection：strict 本地结果使用 train split 内 validation 做 checkpoint selection；`test_as_validation` 只能标记为 upper-bound
- claim status：缺官方 AE/fusion pretrained 权重、official exact test packaging、官方环境和官方完整训练搜索流程时，只能写 `local substitute`、`local strict-validation`、`upper-bound` 或 `blocked official reproduction`

当前推荐命令：

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --train-scenes 32 33 34 \
  --eval-scenes 31 32 33 34 \
  --selection-split validation \
  --fusion-val-fraction 0.1 \
  --gps-feature-mode paper_distance_angle \
  --target-beam-source current \
  --output-root outputs/scenegroup_s32_s34/beambench_image_ae_gps_direct_tableiii/beambench_aligned
```

当前状态：official reproduction 仍为 blocked；本 change 不提升任何旧 `future` target 数值为当前结果。旧 `--target-beam-source future`、旧 GPS 公式、旧 AE 维度、scene31-only、dry-run、mock 和 `test_as_validation` 记录全部是 historical ablation、smoke/mock 或 upper-bound。结果 provenance 统一见 `docs/result_claims_registry.md`，当前主线入口见 `docs/mainline_model_catalog.md` 和 `README_REPRODUCE.md`。

## 审计摘要

- 当前仓库 commit：`00b693b9b0aa42e213884bdf3ddf36a4a25c70f8`
- 官方仓库：`https://github.com/ITU-AI-ML-in-5G-Challenge/BeamBench`
- 官方临时 clone：`/tmp/beambench-official`
- 官方 commit：`8e2c29a2afc898a69b9f9f7ece039d1e48ba60e8`
- 官方 README 推荐命令：`python3 challenge.py --gpu_id 0 --data_folder ./raw_data/test/ --csv ml_challenge_test_multi_modal.csv`
- 官方默认模型目录：`results/models`
- 官方默认预测目录：`results/topk`

已审计官方文件：

- `README.md`
- `Dockerfile`
- `challenge.py`
- `challenge_lstm.py`
- `classical.py`
- `config/camera_ae.cfg`
- `config/gps_dense.cfg`
- `libraries/general.py`
- `models/ae_camera_model.py`
- `models/dense_model.py`

官方仓库缺失或只提供 `.pyc` 的入口依赖包括 `ae_lidar_model.py`、`ae_radar_model.py`、`cl_camera_model.py`、`cl_radar_model.py`、`lstm_model.py`、`mmWave_camera_model.py`、`mmWave_lidar_model.py`、`mmWave_radar_model.py` 以及若干 `config/*.cfg`。因此当前不能把官方原样 `challenge.py` 标记为真实复现完成。

## baseline 覆盖

## 纠偏：论文 Image AE + GPS Direct + Fusion Yes

用户指定的目标不是“所有 BeamBench baseline”，而是 Arnold22 BeamBench Table III 中这一行：

| Camera | Radar | Lidar | GPS | Fusion | Scene 31 | Scene 32 | Scene 33 | Scene 34 | Overall |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| AE | none | none | Direct | Yes | 0.6731 | 0.6173 | 0.8171 | 0.7313 | 0.7127 |

当前项目具备可替代数据源：`dataset/DeepSense6G/scenario31`、`scenario32`、`scenario33`、`scenario34`，并且本地 CSV 含 `camera1..camera8`、`gps1..gps8`、`bs_gps1..bs_gps8`、`beam1..beam8` 和 `future_beam1..future_beam3`。这可以支撑项目内替代实验。

精确官方复现仍缺：

- 官方 Table III 对应训练好的 camera AE/fusion 权重
- 官方匹配缓存 `deepsense_6g_test_match.hdf5` 的完整训练/测试生成流程与权重
- 官方缺失的部分源码/配置，尤其是多模态组合中 `challenge.py` 引用但未随 `.py` 发布的模块
- 严格官方 Python 3.7/CUDA 11.4/PyTorch cu113 环境

已新增项目内本地训练实现：

- 模型与训练闭环：`src/kd_sensing/baselines/beambench/image_ae_gps.py`
- CLI：`src/kd_sensing/cli/train_beambench_image_ae_gps.py`
- script：`scripts/train_beambench_image_ae_gps.py`
- 配置：`configs/fusion/beambench_image_ae_gps_direct.yaml`

该实现直接读取本地 DeepSense6G scene sequence CSV，先训练或加载本仓库 `CameraAutoEncoder`，再冻结 AE encoder，将 image AE latent 与 GPS direct feature concat fusion 到 64-beam classifier，并输出 BeamBench DBA/top-k metrics。它可用于本地 Scenes 31-34 对照实验；若未使用官方权重和官方完整训练搜索流程，不能声称复现官方 Table III 数值。

2026-06-12 口径修订：严格 Arnold22 Table III 使用当前 beam selection，项目命令必须设置 `--target-beam-source current` 或继承 `configs/fusion/beambench_image_ae_gps_direct.yaml` 的 current 默认值。下文早期带 `--target-beam-source future` 的命令和数值仅保留为历史本地 sequence horizon ablation，不再作为 Table III strict setup 或推荐结果。

建议命令：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml

conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --scene 32
```

## 3090 多核服务器训练加速

已为专用 Image AE + GPS Direct 入口加入默认加速路径：

- 冻结 AE latent cache：AE checkpoint 确定后，将 train/test camera AE latent 写入当前 run 的 `feature_cache/*.pt`，fusion 阶段不再每个 epoch 反复读图和运行 AE encoder。
- DataLoader 并行：默认 `num_workers=8`、`pin_memory=true`、`persistent_workers=true`、`prefetch_factor=2`，正式训练可通过 `--num-workers` 按服务器 CPU 核数提高。
- CUDA 加速：默认启用 AMP + GradScaler、TF32、cuDNN benchmark、fused AdamW 和 non-blocking transfer。
- 精度边界：不减少样本数、不降低 `ae_image_size=64`、不减少 `ae_epochs=20` / `fusion_epochs=80` 上限、不改变 early stopping 或 DBA 选 best；cache 只在 AE frozen 时启用，语义等价于先算好 frozen image latent。

推荐正式命令：

```bash
conda run -n kd_mm_beam kd-sensing-train-beambench-image-ae-gps \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --num-workers 12 \
  --ae-batch-size 128 \
  --fusion-batch-size 512
```

调试完全 fp32/在线 forward：

```bash
conda run -n kd_mm_beam kd-sensing-train-beambench-image-ae-gps \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --no-feature-cache \
  --no-amp \
  --no-tf32
```

| baseline | type_list | modalities | status | reason |
|---|---|---|---|---|
| Camera AE + GPS | `camera_ae_gps` | camera, gps | blocked | 缺官方数据和 checkpoint |
| GPS dense | `gps_dense` | gps | blocked | 缺官方 GPS dense checkpoint 和真实 CSV |
| late fusion pretrained features | `radar_dense_camera_ae_gps` | radar, camera, gps | blocked | 缺 fusion checkpoint、真实 CSV；部分可选源码缺失 |
| LiDAR + Camera + Radar + GPS | `lidar_ae_camera_ae_radar_cl_gps` | lidar, camera, radar, gps | blocked | 官方 LiDAR/Radar/CL/LSTM 等源码不完整，权重缺失 |
| MOCK smoke | `mock_tiny_mlp` | camera, lidar, radar, gps | completed mock | 仅验证代码路径，不能作为真实结果 |

## 本地 Image AE + GPS Direct dry-run

Command：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py --config configs/fusion/beambench_image_ae_gps_direct.yaml --dry-run --output-dir outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast
```

结果：

- run type：local real-data dry-run，非官方权重
- scene：31
- train CSV：`dataset/DeepSense6G/scenario31/train_seqs_RA_GPS_LIDAR.csv`
- test CSV：`dataset/DeepSense6G/scenario31/test_seqs_RA_GPS_LIDAR.csv`
- train/test sample count：4 / 4
- AE epoch / fusion epoch：1 / 1
- AE checkpoint：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/camera_ae/checkpoints/best.pt`
- fusion checkpoint：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/checkpoints/best_image_ae_gps_direct.pt`
- feature cache：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/feature_cache/train_camera_ae_latents.pt` 和 `test_camera_ae_latents.pt`
- predictions：`outputs/beambench_image_ae_gps_direct/dry_run_scene31_fast/predictions.csv`
- acceleration：CUDA AMP/TF32 enabled，feature cache active，dry-run worker 强制为 0
- valid label count：4
- `official_top3_dba`: 0.0
- `circular_top3_dba`: 0.0

这些数值只证明真实本地数据训练闭环可运行；样本数和 epoch 均为 dry-run，不能作为论文结果。

## Historical log：本地 Image AE + GPS Direct scene31 完整运行

状态：historical scene31-only local run。该记录使用单场景 local sequence split，只能解释早期训练闭环和 scene31 专项现象，不作为当前 Table III strict setup 或当前推荐结果。

Command 由用户在本地完成，输出目录为：

```text
outputs/beambench_image_ae_gps_direct_scene31
```

运行摘要：

- run type：real local DeepSense6G scene31 sequence split，非官方权重
- train/test sample count：5049 / 1443
- AE checkpoint：`outputs/beambench_image_ae_gps_direct_scene31/camera_ae/checkpoints/best.pt`
- fusion checkpoint：`outputs/beambench_image_ae_gps_direct_scene31/checkpoints/best_image_ae_gps_direct.pt`
- predictions：`outputs/beambench_image_ae_gps_direct_scene31/predictions.csv`
- feature cache：`outputs/beambench_image_ae_gps_direct_scene31/feature_cache/train_camera_ae_latents.pt` 和 `test_camera_ae_latents.pt`
- acceleration：CUDA AMP/TF32 enabled，frozen AE latent cache active，`num_workers=12`
- selected checkpoint：epoch 35 by best `official_top3_dba`
- fusion epochs run：50

指标：

| field | value |
|---|---:|
| official_top1_acc | 0.4435204435204435 |
| official_top3_acc | 0.8205128205128205 |
| official_top5_acc | 0.9196119196119196 |
| official_top3_dba | 0.8676830676830677 |
| circular_top3_dba | 0.9194733194733195 |

训练解读：AE 重构验证 loss 从 `0.01456` 降到 `0.0009905`，说明 camera AE 预训练正常；fusion DBA 在前 10 epoch 快速提升到 `0.8466`，epoch 35 达到最佳 `0.8677`，之后 train loss 继续下降但 DBA 基本平台化，属于轻微过拟合/饱和，early stopping 选择 best checkpoint 是合理的。该结果仍不能直接与论文 Table III scene31 `0.6731` 做数值高低比较，因为本运行使用本地 sequence CSV split 和本仓库训练流程，而非官方 pretrained 权重、官方完整搜索流程和官方 challenge test packaging。

## Historical log：本地论文 split，scenes 32-34 联合训练，scenes 31-34 测试

用户纠正后，Table III 复现协议改为：在 scenes 32、33、34 上联合训练一个 Camera AE+GPS Direct fusion classifier，并在 scenes 31、32、33、34 上评估同一个 best checkpoint。论文目标行为：

| Scene 31 | Scene 32 | Scene 33 | Scene 34 | Overall |
|---:|---:|---:|---:|---:|
| 0.6731 | 0.6173 | 0.8171 | 0.7313 | 0.7127 |

已新增 joint runner：

状态：historical sequence-prediction ablation。下面的旧命令包含 `--target-beam-source future`，不得作为当前 Table III strict setup 或推荐结果；当前命令见本文开头 Current Summary。

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --train-scenes 32 33 34 \
  --eval-scenes 31 32 33 34 \
  --selection-split validation \
  --fusion-val-fraction 0.1 \
  --gps-feature-mode paper_distance_angle \
  --target-beam-source future \
  --num-workers 12 \
  --ae-batch-size 128 \
  --fusion-batch-size 512 \
  --feature-cache-batch-size 256
```

`paper_distance_angle` 对齐官方 `challenge.py` 的 GPS Direct 输入，即 `[distance, calibrated_angle_deg]` 二维特征；`paper_calibrated_relative_polar` 是三维 `[distance, sin(theta), cos(theta)]` ablation。

### Historical ablation：Scene31 泛化专项修复

用户进一步收窄目标为优先提升 scene31 泛化。复核官方 `challenge.py` 后发现两处影响 scene31 的关键差异：

- `paper_distance_angle` 角度必须使用官方 `arctan(x/y)`，不是 `atan2(x, y)`；否则 scene31/34 会跨 `±180` 断点。
- scene32 的官方校准角是 `-0.8125375604986421 + pi/2 = 0.7583`，不是早期误用的 `-0.76`；误用会让 scene32 角度分布异常发散。

只修 GPS 公式和 scene32 校准角、仍复用旧 128 维/64px AE 时，scene31 从 strict validation 旧结果 `0.3786` 提升到：

| AE | GPS 修复 | Scene31 local DBA | Paper Scene31 DBA | Delta |
|---|---|---:|---:|---:|
| 128d/64px reused | yes | 0.5569 | 0.6731 | -0.1162 |

进一步按官方 Camera AE 512 维输出方向重新训练 512 维 AE 后，scene31 达到：

| AE | GPS 修复 | Scene31 local DBA | Paper Scene31 DBA | Delta |
|---|---|---:|---:|---:|
| 512d/64px retrained | yes | 0.6824 | 0.6731 | +0.0093 |

对应命令：

状态：historical scene31-only sequence-prediction ablation。下面旧命令包含 `--target-beam-source future`，不得作为当前 Table III strict setup 或当前推荐结果。

```bash
conda run -n kd_mm_beam kd-sensing-run-beambench-image-ae-gps-tableiii \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --train-scenes 32 33 34 \
  --eval-scenes 31 \
  --output-root outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_ae512_validation \
  --selection-split validation \
  --fusion-val-fraction 0.1 \
  --gps-feature-mode paper_distance_angle \
  --target-beam-source future \
  --num-workers 12 \
  --ae-batch-size 128 \
  --fusion-batch-size 512 \
  --feature-cache-batch-size 256
```

该结果只说明 scene31 单项已接近并略高于论文 Table III 的 scene31 数值；按用户要求，暂未重新优化 scenes 32-34 和 overall。

### Historical ablation：31-34 完整 eval-only strict-validation checkpoint

随后按用户要求重新追 scenes 32-34 和 overall。使用同一个 strict validation checkpoint：

```text
outputs/beambench_image_ae_gps_direct_tableiii/scene31_gpsfix_ae512_validation/checkpoints/best_image_ae_gps_direct_paper_split.pt
```

直接 eval-only 到 scenes 31-34，输出目录：

```text
outputs/beambench_image_ae_gps_direct_tableiii/full_gpsfix_ae512_validation_checkpoint_eval
```

结果如下：

| Scene | Local DBA | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.6824 | 0.6731 | +0.0093 |
| 32 | 0.7431 | 0.6173 | +0.1258 |
| 33 | 0.8371 | 0.8171 | +0.0200 |
| 34 | 0.8158 | 0.7313 | +0.0845 |
| weighted overall | 0.7594 | 0.7127 | +0.0467 |

这组是历史 future-target local strict-validation 结果：同一个 32-34 训练出的 checkpoint，在 31-34 四个场景上均超过论文表中对应 DBA。它仍使用本地 sequence split 和本仓库 AE/fusion 训练流程，不是官方 unseen test packaging；在 current-target 口径确立后，不得作为当前 Table III substitute 主结论。

### Historical ablation：31-34 完整 retrain，strict validation 与 upper-bound

重新训练 fusion 并评估 31-34 的 strict validation 结果为：

| Scene | Local DBA | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.6594 | 0.6731 | -0.0137 |
| 32 | 0.7879 | 0.6173 | +0.1706 |
| 33 | 0.8471 | 0.8171 | +0.0300 |
| 34 | 0.8134 | 0.7313 | +0.0821 |
| weighted overall | 0.7626 | 0.7127 | +0.0499 |

`test_as_validation` 本地 upper-bound 结果为：

| Scene | Local DBA | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.6756 | 0.6731 | +0.0025 |
| 32 | 0.8095 | 0.6173 | +0.1922 |
| 33 | 0.8414 | 0.8171 | +0.0243 |
| 34 | 0.8296 | 0.7313 | +0.0983 |
| weighted overall | 0.7745 | 0.7127 | +0.0618 |

upper-bound 使用 test CSV 选 checkpoint，只用于查看本地上限，不作为官方 unseen evaluation。

### Historical strict validation result

状态：historical sequence-prediction ablation。输出目录：`outputs/beambench_image_ae_gps_direct_tableiii/paper_split_official_gps_future_validation`

| Scene | Local DBA | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.3786 | 0.6731 | -0.2945 |
| 32 | 0.5031 | 0.6173 | -0.1142 |
| 33 | 0.7587 | 0.8171 | -0.0584 |
| 34 | 0.8045 | 0.7313 | +0.0732 |
| weighted overall | 0.5820 | 0.7127 | -0.1307 |

### Historical local upper-bound result

状态：historical sequence-prediction ablation + upper-bound。输出目录：`outputs/beambench_image_ae_gps_direct_tableiii/paper_split_official_gps_future_test_as_validation`

| Scene | Local DBA | Paper DBA | Delta |
|---:|---:|---:|---:|
| 31 | 0.4295 | 0.6731 | -0.2436 |
| 32 | 0.6370 | 0.6173 | +0.0197 |
| 33 | 0.7865 | 0.8171 | -0.0306 |
| 34 | 0.7731 | 0.7313 | +0.0418 |
| weighted overall | 0.6282 | 0.7127 | -0.0845 |

该 upper-bound 使用 `test_as_validation` 选择 best checkpoint，可帮助观察本地训练最多接近到哪里，但不是官方 unseen test 口径。

### Ablation observations

以下 ablation 记录来自 2026-06-12 口径修订前的历史本地运行；其中 `future` target 结果不得再作为 Table III strict setup 解读。

- 三维 `paper_calibrated_relative_polar` + strict validation：weighted overall `0.5976`，Scene31 `0.2435`，Scene32/33/34 分别为 `0.7236/0.8295/0.8029`。
- 三维 `paper_calibrated_relative_polar` + `test_as_validation` upper-bound：weighted overall `0.5760`，Scene31 `0.3662`。
- `target_beam_source=current` ablation：weighted overall `0.5249`，Scene31 仅 `0.0679`，明显不适合作为本地 Table III 主口径。

结论更新：纠正联合训练协议后，先前 scene31 单场景 `0.8677` 确认是不可比的乐观结果。旧四场景 future-target 结果的主要差距来自 Scene31 未见分布；在修复官方 GPS 角度公式、scene32 校准角并使用 512 维 AE 后，historical local strict-validation checkpoint 的 scenes 31-34 全部超过论文对应 DBA，weighted overall 为 `0.7594`。该结果仍是本地 sequence split 复现，不等同官方 unseen test packaging，也不再覆盖 current-target summary。

## metric 口径

- `official_top*_acc`：非环形 top-k 命中率，使用 BeamBench/DeepSense top-k beam id 语义
- `official_top3_dba`：对照官方 `libraries/general.py::compute_DBA_score`，非环形 absolute beam distance，`delta=5`
- `circular_top*_acc` 与 `circular_top3_dba`：本仓库 64-beam circular 口径，用于后续 residual/BGAM/fusion 工作流
- 本次 mock 使用 `beam_shift=0`；真实官方提交 CSV 通常需要记录 `beam_shift=1`

不同 DBA 口径使用不同字段名，不能混用为同一指标。

## 本次 mock checker

Command：

```bash
conda run -n kd_mm_beam python scripts/check_dataset.py --data-root outputs/beambench_baseline/mock_dataset --csv ml_challenge_mock_multi_modal.csv --scene MOCK --num-beams 64 --beam-shift 0 --output outputs/beambench_baseline/mock_dataset/check_report.json
```

结果：

- `mock_data: true`
- row count：12
- camera/LiDAR/radar/GPS missing count：0
- label count：24，包括 numeric `label` 和 `future_beam1` 文件
- invalid label count：0
- scene/sample/sequence/timestamp：均可解析
- report path：`outputs/beambench_baseline/mock_dataset/check_report.json`

## 本次 mock baseline smoke

Command：

```bash
conda run -n kd_mm_beam python scripts/train_baseline.py --mock --epochs 2 --num-beams 64 --data-root outputs/beambench_baseline/mock_dataset --csv ml_challenge_mock_multi_modal.csv --output-dir outputs/beambench_baseline/mock_smoke --device cpu
```

结果：

- `mock_data: true`
- checkpoint：`outputs/beambench_baseline/mock_smoke/checkpoints/mock_beambench_baseline.pt`
- logits：`outputs/beambench_baseline/mock_smoke/mock_logits.npy`
- log：`outputs/beambench_baseline/mock_smoke/train_log.json`
- sample count：12
- valid label count：12
- `official_top1_acc`: 0.3333333333333333
- `official_top3_acc`: 0.8333333333333334
- `official_top5_acc`: 0.9166666666666666
- `official_top3_dba`: 0.638888888888889
- `circular_top1_acc`: 0.3333333432674408
- `circular_top3_acc`: 0.8333333134651184
- `circular_top5_acc`: 0.9166666865348816
- `circular_top3_dba`: 0.8333333333333334

这些数值只证明 mock 代码路径闭环，不能作为真实 BeamBench 指标。

## 官方 eval wrapper plan

Command：

```bash
conda run -n kd_mm_beam python scripts/eval_baseline.py --official-root /tmp/beambench-official --data-root dataset/DeepSense6G/raw_data/test --csv ml_challenge_test_multi_modal.csv --type-list radar_dense_camera_ae_gps --output-dir outputs/beambench_baseline/eval
```

结果：blocked，非零退出符合预期。

主要 blocked reasons：

- `dataset/DeepSense6G/raw_data/test/ml_challenge_test_multi_modal.csv` 不存在
- `/tmp/beambench-official/results/models/adapt_bb_camera_ae_0_42.pth` 不存在
- `/tmp/beambench-official/results/models/fusion_adapt_radar_dense_camera_ae_gps_0_42.pth` 不存在
- 官方仓库缺失 `models/ae_lidar_model.py`、`models/ae_radar_model.py`、`models/cl_camera_model.py`、`models/cl_radar_model.py`、`models/lstm_model.py`、`models/mmWave_*`
- 官方仓库缺失 `config/camera_cl.cfg`、`config/radar_ae.cfg`、`config/radar_cl.cfg`、`config/lidar_ae.cfg`、`config/lstm_model.cfg`、`config/*_mmWave.cfg`

真实结果栏保持 blocked，不填任何论文指标或 leaderboard 指标。
