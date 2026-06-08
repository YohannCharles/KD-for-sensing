# BeamBench baseline 复现说明

本文件记录 `reproduce-beambench-baseline` 的可执行 workflow。所有项目相关 Python 命令都使用 `kd_mm_beam` 环境。

## 1. 环境检查

```bash
conda run -n kd_mm_beam python -c "import sys, torch; import torchvision; print(sys.version); print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

当前已采集环境写入 `ENVIRONMENT.md`。官方 BeamBench README/Dockerfile 目标环境是 Ubuntu 18.04、CUDA 11.4、Python 3.7 和 PyTorch cu113 wheel；当前 `kd_mm_beam` 是最小可运行方案，不等同于官方原始环境。

## 2. 数据检查

真实 BeamBench/DeepSense6G 数据到位后，先运行只读 checker：

```bash
conda run -n kd_mm_beam python scripts/check_dataset.py \
  --data-root dataset/DeepSense6G/raw_data/test \
  --csv ml_challenge_test_multi_modal.csv \
  --scene 31-34 \
  --num-beams 64 \
  --beam-shift 1 \
  --output outputs/beambench_baseline/real_data_check.json
```

checker 会报告 CSV 字段、camera/LiDAR/radar/GPS 路径引用、label 范围、scene/sample/sequence/timestamp 标识和缺失比例，不移动、不删除、不生成真实数据。

## 3. mock smoke

真实数据或官方权重不可用时，只能运行显式 `MOCK` smoke：

```bash
conda run -n kd_mm_beam python scripts/train_baseline.py \
  --mock \
  --epochs 2 \
  --num-beams 64 \
  --data-root outputs/beambench_baseline/mock_dataset \
  --csv ml_challenge_mock_multi_modal.csv \
  --output-dir outputs/beambench_baseline/mock_smoke \
  --device cpu
```

单独检查 mock dataset：

```bash
conda run -n kd_mm_beam python scripts/check_dataset.py \
  --data-root outputs/beambench_baseline/mock_dataset \
  --csv ml_challenge_mock_multi_modal.csv \
  --scene MOCK \
  --num-beams 64 \
  --beam-shift 0 \
  --output outputs/beambench_baseline/mock_dataset/check_report.json
```

mock 输出包含 `mock_data: true`，不得用于论文结果、leaderboard 或真实 BeamBench baseline 对比。

## 4. 论文 Image AE + GPS Direct 目标行

Arnold22 BeamBench Table III 中用户关心的是：

```text
Camera=AE, Radar=none, Lidar=none, GPS=Direct, Fusion=Yes
Scene31=0.6731, Scene32=0.6173, Scene33=0.8171, Scene34=0.7313, Overall=0.7127
```

本仓库没有官方 pretrained camera AE/fusion 权重，因此不能直接声称复现这组数值。当前可替代数据是本地 `dataset/DeepSense6G/scenario31` 到 `scenario34`，可用配置是：

```text
configs/fusion/beambench_image_ae_gps_direct.yaml
```

推荐直接运行论文 row 专用入口。该入口会先训练或加载本仓库 Camera AE checkpoint，然后冻结 AE encoder 训练 GPS direct fusion classifier：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml
```

等价 console script：

```bash
conda run -n kd_mm_beam kd-sensing-train-beambench-image-ae-gps \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml
```

默认配置面向 RTX 3090 + 多核 CPU：启用冻结 AE latent cache、CUDA AMP、TF32、fused AdamW、pin memory、persistent workers、prefetch 和 non-blocking transfer；不减少样本数、epoch、image size 或 DBA 选 best 逻辑。若服务器 CPU 核心更多，可直接提高 worker：

```bash
conda run -n kd_mm_beam kd-sensing-train-beambench-image-ae-gps \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --num-workers 12 \
  --ae-batch-size 128 \
  --fusion-batch-size 512 \
  --feature-cache-batch-size 256
```

排查精度或调试完全在线 forward 时，可关闭加速项：

```bash
conda run -n kd_mm_beam kd-sensing-train-beambench-image-ae-gps \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --no-feature-cache \
  --no-amp \
  --no-tf32
```

其它场景使用 override：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --scene 34
```

快速 smoke：

```bash
conda run -n kd_mm_beam python scripts/train_beambench_image_ae_gps.py \
  --config configs/fusion/beambench_image_ae_gps_direct.yaml \
  --dry-run \
  --output-dir outputs/beambench_image_ae_gps_direct/dry_run_scene31
```

该实现是项目内本地训练复现路径，不依赖官方预训练权重；若没有官方权重和官方完整训练搜索流程，不能把本地数值声称为官方 Table III 数值。

论文 Table III 对齐实验应使用 joint split runner，而不是逐场景分别训练：

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

其中 `paper_distance_angle` 对齐官方 `challenge.py` 的 GPS Direct 输入 `[distance, calibrated_angle_deg]`。若要查看本地上限，可把 `--selection-split validation --fusion-val-fraction 0.1` 改为 `--selection-split test_as_validation`，但该口径不等同官方 unseen test。

## 5. 官方评估 wrapper

官方仓库临时 clone 位置：

```text
/tmp/beambench-official
```

生成官方评估计划并记录 blocked reason：

```bash
conda run -n kd_mm_beam python scripts/eval_baseline.py \
  --official-root /tmp/beambench-official \
  --data-root dataset/DeepSense6G/raw_data/test \
  --csv ml_challenge_test_multi_modal.csv \
  --type-list radar_dense_camera_ae_gps \
  --output-dir outputs/beambench_baseline/eval
```

只有在官方数据、权重、源码/配置和兼容环境齐备时，才显式增加 `--execute` 运行 `challenge.py`：

```bash
conda run -n kd_mm_beam python scripts/eval_baseline.py \
  --official-root /tmp/beambench-official \
  --data-root /path/to/raw_data/test \
  --csv ml_challenge_test_multi_modal.csv \
  --type-list radar_dense_camera_ae_gps \
  --output-dir outputs/beambench_baseline/eval \
  --execute
```

## 6. 报告

本次实现维护这些文档：

- `ENVIRONMENT.md`
- `DATASET_STRUCTURE.md`
- `BASELINE_REPORT.md`
- `PATCH_NOTES.md`
- `TODO_FOR_ATTENTION_MODULE.md`
- `results/reproduce_baseline.md`

运行后请把实际 command、commit、dataset split、modalities、checkpoint path、metrics、日志路径和 mock/real 标记补入 `BASELINE_REPORT.md` 与 `results/reproduce_baseline.md`。真实复现不可用时必须写 blocked，不能填虚假指标。
