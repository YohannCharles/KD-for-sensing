# BeamBench / DeepSense6G 数据结构说明

## 官方 BeamBench 期望结构

官方 `challenge.py` 默认读取：

```text
raw_data/test/
  ml_challenge_test_multi_modal.csv
  unit1/...
  unit2/...
```

CSV 中官方脚本直接使用的字段包括：

- camera：`unit1_rgb_5`
- radar：`unit1_radar_5`
- LiDAR：`unit1_lidar_5`
- BS GPS：`unit1_loc`
- UE GPS：`unit2_loc_1`、`unit2_loc_2`

官方评估脚本会按 `data_folder + csv` 读取 CSV，再按 `data_folder` 解析各行相对路径。GPS 文件通过 `numpy.loadtxt` 读取经纬度，radar 通过 `numpy.load` 读取，camera 通过 image reader 读取，LiDAR 通过 Open3D 读取 point cloud。

## 本仓库 sequence CSV 等价字段

本仓库 DeepSense6G sequence CSV 常见字段包括：

- camera：`camera1`、`camera2` 或官方兼容的 `unit1_rgb_*`
- radar：`radar1`、`radar2` 或 `unit1_radar_*`
- LiDAR：`lidar1`、`lidar2` 或 `unit1_lidar_*`
- GPS：`gps*`、`bs_gps*`、`future_gps*`、`future_bs_gps*`、`unit1_loc`、`unit2_loc_*`
- label 文件：`beam*`、`future_beam*`
- numeric label：`label`、`beam_label`、`target_label`、`target_beam`、`future_beam_label*`
- 标识：`scene`、`scene_id`、`scenario`、`sample`、`sample_id`、`seq`、`seq_index`、`timestamp`

`scripts/check_dataset.py` 同时支持官方字段和这些等价字段。

## label / beam index

官方 `challenge.py` 默认 `--beams_shift 1`，因为官方提交 CSV 期望 1..64 beam id。模型内部和本仓库多数 metric helper 使用 0..63 类别。检查真实官方 CSV 时建议：

```bash
conda run -n kd_mm_beam python scripts/check_dataset.py \
  --data-root <raw_data/test> \
  --csv ml_challenge_test_multi_modal.csv \
  --num-beams 64 \
  --beam-shift 1
```

如果 CSV 已是 0-based label，则使用 `--beam-shift 0`。

## Scenes 31-34 放置建议

本仓库 README 中 DeepSense6G 默认根目录是 `dataset/DeepSense6G/scenario31`。BeamBench 官方 raw test layout 可独立放置，例如：

```text
dataset/DeepSense6G/raw_data/test/ml_challenge_test_multi_modal.csv
dataset/DeepSense6G/scenario31/
dataset/DeepSense6G/scenario32/
dataset/DeepSense6G/scenario33/
dataset/DeepSense6G/scenario34/
```

真实数据属于本地产物，默认不提交。checker 只读扫描路径引用和 label 范围，不移动、不删除、不重写数据。

## mock dataset

`scripts/train_baseline.py --mock` 会在 ignored 的 `outputs/beambench_baseline/mock_dataset/` 生成极小 mock CSV 和占位 camera/LiDAR/radar/GPS/beam label 输入。每个 artifact 都带 `MOCK` 或 `mock_data: true` 标记，仅用于验证 dataloader、forward、loss、metric、checkpoint save/load 和 evaluation 代码路径。
