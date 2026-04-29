# KD for Sensing

本仓库现已整理为可安装的 `src/kd_sensing` 包，并提供基于配置文件的训练、评估和预处理入口。

## 安装

```bash
conda activate kd_mm_beam
pip install -e .
```

包导入过程不会产生副作用：

```bash
python -c "import kd_sensing"
```

## 目录结构

```text
configs/
  image/          # 仅图像模型：无 KD、logits KD、RKD 配置
  radar/          # 仅雷达模型：无 KD、logits KD、RKD 配置
  gps/            # 仅 GPS/position 模型：无 KD、KD 和 GPS-Rel-Polar 配置
  lidar/          # 仅 LiDAR BEV 模型：无 KD、logits KD、RKD 配置
  mmwave/         # 仅 mmWave power vector 模型：无 KD、logits KD、RKD 配置
  fusion/         # 可选 image/radar/gps/lidar/mmwave 融合模型配置
  preprocess/     # CSV、雷达、序列预处理配置
scripts/
  train.py
  evaluate.py
  preprocess.py
src/kd_sensing/
  cli/
  config/
  data/
  distillation/
  engine/
  evaluation/
  models/
  preprocessing/
  utils/
```

大型数据和预训练权重继续保留在原有位置：

- `dataset/`
- `All_models/`

配置文件中的相对路径会从项目根目录解析，因此可以在子目录中启动命令。

## 训练

推荐按 `teacher_no_kd -> student_no_kd -> logits_kd/rkd` 的顺序运行。Image 与 image+radar
兼容 KD 配置默认读取随附 `All_models` 中的一层 image teacher 或二层 image+radar teacher
权重；radar/GPS/LiDAR 单模态 KD 配置会优先从最佳 checkpoint registry 读取同模态
teacher no-KD 的最高验证 Top-1 权重，缺失时再回退到
`outputs/<slug>_teacher_no_kd/checkpoints/best.pth`。也可以通过 `paths.weights_dir` 和
`distillation.teacher_model_name` 覆盖旧式 teacher checkpoint 路径；绝对路径和评估入口
`--weights` 始终优先于 registry。

单模态 canonical 配置矩阵：

| 模态 | Teacher baseline | Student baseline | KD |
| --- | --- | --- | --- |
| image | `configs/image/teacher_no_kd.yaml` | `configs/image/student_no_kd.yaml` | `configs/image/logits_kd.yaml`, `configs/image/rkd.yaml` |
| radar | `configs/radar/teacher_no_kd.yaml` | `configs/radar/student_no_kd.yaml` | `configs/radar/logits_kd.yaml`, `configs/radar/rkd.yaml` |
| gps | `configs/gps/teacher_no_kd.yaml` | `configs/gps/student_no_kd.yaml` | `configs/gps/logits_kd.yaml`, `configs/gps/rkd.yaml` |
| lidar | `configs/lidar/teacher_no_kd.yaml` | `configs/lidar/student_no_kd.yaml` | `configs/lidar/logits_kd.yaml`, `configs/lidar/rkd.yaml` |
| mmwave | `configs/mmwave/teacher_no_kd.yaml` | `configs/mmwave/student_no_kd.yaml` | `configs/mmwave/logits_kd.yaml`, `configs/mmwave/rkd.yaml` |

```bash
python scripts/train.py --config configs/image/teacher_no_kd.yaml
python scripts/train.py --config configs/image/student_no_kd.yaml
python scripts/train.py --config configs/image/logits_kd.yaml
python scripts/train.py --config configs/image/rkd.yaml
```

Fusion canonical 配置覆盖固定顺序 `image -> radar -> gps -> lidar -> mmwave` 下的 26 个多模态 slug：

```text
image_radar, image_gps, image_lidar, radar_gps, radar_lidar, gps_lidar
image_radar_gps, image_radar_lidar, image_gps_lidar, radar_gps_lidar
image_radar_gps_lidar

以及包含 mmwave 的所有双模态、三模态、四模态和五模态组合，例如
image_mmwave, radar_mmwave, gps_mmwave, lidar_mmwave, image_radar_mmwave,
image_radar_gps_lidar_mmwave
```

每个 slug 都有四个配置：

```bash
python scripts/train.py --config configs/fusion/<slug>_teacher_no_kd.yaml
python scripts/train.py --config configs/fusion/<slug>_student_no_kd.yaml
python scripts/train.py --config configs/fusion/<slug>_logits_kd.yaml
python scripts/train.py --config configs/fusion/<slug>_rkd.yaml
```

例如 image+radar：

```bash
python scripts/train.py --config configs/fusion/image_radar_teacher_no_kd.yaml
python scripts/train.py --config configs/fusion/image_radar_student_no_kd.yaml
python scripts/train.py --config configs/fusion/image_radar_logits_kd.yaml
python scripts/train.py --config configs/fusion/image_radar_rkd.yaml
```

单模态配置统一使用 `gru_params: [64, 64, 1]`。其中 image 参数来自上游 image 单模态脚本和
`All_models/params_Image*.txt`，radar/GPS/LiDAR/mmWave 是本项目新增单模态，默认继承 image 同角色
配置参数。Image+radar fusion 按上游 `train_both.py` 和 `All_models/params_Both*.txt` 保持
teacher 二层 GRU、student 一层 GRU；其它 fusion 组合属于扩展配置，不声明为原论文结果复现。
Checkpoint 加载默认严格校验 missing/unexpected keys，结构不匹配会直接报错；需要兼容性调试时可显式设置
`checkpoint.strict_load=false`。

训练会将当前配置最高验证 Top-1 checkpoint 复制到默认 registry 目录
`outputs/best_checkpoints/`，文件名包含配置 slug、角色/KD 模式和精度，例如
`gps_teacher_no_kd_acc_0.8123.pth`。同名 checkpoint 会写入 `.json` sidecar，记录源运行目录、
epoch、split 样本数、加载来源和 GPS/LiDAR 归一化工件。可通过
`checkpoint.registry.dir`、`checkpoint.registry.enabled` 和 `checkpoint.registry.prefer` 调整目录、
开关和加载优先级。

legacy 入口继续保留，但新实验优先使用上面的 canonical 名称：

| Legacy 配置 | 当前语义 | 推荐 canonical 入口 |
| --- | --- | --- |
| `configs/image/no_kd.yaml` | image lightweight student no-KD | `configs/image/student_no_kd.yaml` |
| `configs/radar/no_kd.yaml` | radar teacher no-KD | `configs/radar/teacher_no_kd.yaml` |
| `configs/gps/no_kd.yaml` | GPS teacher no-KD | `configs/gps/teacher_no_kd.yaml` |
| `configs/lidar/no_kd.yaml` | LiDAR teacher no-KD | `configs/lidar/teacher_no_kd.yaml` |
| `configs/fusion/no_kd.yaml` | image+radar student no-KD | `configs/fusion/image_radar_student_no_kd.yaml` |
| `configs/fusion/logits_kd.yaml`, `configs/fusion/rkd.yaml` | image+radar legacy KD/pretrained 入口 | `configs/fusion/image_radar_logits_kd.yaml`, `configs/fusion/image_radar_rkd.yaml` |
| `configs/fusion/image_gps_no_kd.yaml` | image+GPS student no-KD | `configs/fusion/image_gps_student_no_kd.yaml` |
| `configs/fusion/radar_gps_no_kd.yaml` | radar+GPS student no-KD | `configs/fusion/radar_gps_student_no_kd.yaml` |
| `configs/fusion/radar_lidar_no_kd.yaml` | radar+LiDAR student no-KD | `configs/fusion/radar_lidar_student_no_kd.yaml` |
| `configs/fusion/all_modalities_no_kd.yaml` | image+radar+GPS student no-KD | `configs/fusion/image_radar_gps_student_no_kd.yaml` |
| `configs/fusion/all_modalities_lidar_no_kd.yaml` | image+radar+GPS+LiDAR student no-KD | `configs/fusion/image_radar_gps_lidar_student_no_kd.yaml` |

Radar-only 配置注册名保持 `radar_teacher` 和 `radar_student`；对应 Python 类名分别为
`RadarModalityNet` 和 `RadarStudentModalityNet`，与 image/GPS 的 `*ModalityNet`
命名风格一致。如果使用自定义 RadarTeacher 权重，可以覆盖路径：

```bash
python scripts/train.py --config configs/radar/logits_kd.yaml \
  --override paths.weights_dir=/path/to/checkpoints \
  --override distillation.teacher_model_name=best.pth
```

GPS-only 配置统一使用 `gps_feature_mode: relative_polar`，即基于 UE-BS 相对 UTM 坐标构造
`[dist, sin_theta, cos_theta]`。六组 GPS 预处理对比后，主路径只保留 GPS-Rel-Polar：

```bash
python scripts/train.py --config configs/gps/ablation_relative_polar.yaml
```

LiDAR-only 配置使用 `lidar1..lidar8` 序列列读取点云，并在线转换成 BEV 伪图像：
默认通道为 height、intensity、density，默认尺寸为 `224x224`，默认 ROI 为
`[-30, 30, -30, 30, -3, 5]`。内置读取器支持 `.mat`、`.npy` 点云数组、ASCII PCD 和
文本/CSV 数值点云；二进制 PCD 需要先离线转换为 ASCII PCD 或 `.npy`。
LiDAR 默认按样本懒加载，不在 Dataset 初始化阶段扫描全训练集计算 z-score；BEV 构造本身会输出
稳定的局部归一化范围。需要全局通道统计时，显式启用流式统计：

```bash
conda run --no-capture-output -n kd_mm_beam python -u scripts/train.py \
  --config configs/lidar/teacher_no_kd.yaml \
  -o data.dataset.lidar_normalization.enabled=true \
  -o data.dataset.lidar_normalization.mode=streaming_stats \
  -o data.dataset.lidar_normalization.stats_path=outputs/cache/lidar_stats_train.npz
```

使用 `conda run` 训练时建议加 `--no-capture-output` 和 `python -u`，否则 tqdm/stderr 可能不会实时显示。

mmWave-only 配置使用 `mmwave1..mmwave8` 序列列读取 `unit1_pwr_60ghz` /
`unit1/mmWave_data/mmWave_power_*.txt` 的 64 维 receive-power vector，先做 finite 清洗和 dB 压缩，
再用训练集 fit 的 `MmWaveStandardScaler` 做 z-score。模型注册名为 `mmwave_teacher` 和
`mmwave_student`，feature extractor 注册名为 `mmwave_feature_extractor`；训练会保存
`artifacts/mmwave_scaler.npz`，评估 registry checkpoint 时会复用该 scaler。

注意：当前默认 mmWave 输入和 beam label 都来自同一个 power vector，因此当前时隙 beam 对 mmWave
接近“答案可见”，未来 3 步更像预测。默认仍保持 8 个历史输入 + 3 步预测，便于跨模态比较；论文或实验表
建议分开报告当前步和未来步，后续可另做只使用 `t-1` 历史的 lagged mmWave 消融。

Fusion 模型通过 `model.teacher.modalities` 和 `model.student.modalities` 选择参与融合的模态，
可用值为 `image`、`radar`、`gps`、`lidar`、`mmwave`。canonical fusion 配置中 teacher/student 的
`modalities` 始终一致；启用 GPS 的配置使用 `gps_feature_mode: relative_polar` 和
`gps_input_size: 3`，启用 LiDAR 的配置使用 BEV 默认字段和 `lidar_channels: 3`，启用 mmWave 的配置使用
`mmwave_input_size: 64` 和 `mmwave_normalize: true`。

可以使用点号分隔的键覆盖配置值：

```bash
python scripts/train.py --config configs/image/rkd.yaml training.epochs=1 data.dataset.portion=0.05
```

当前兼容模型对输入尺寸有结构性限制：image-only 和包含 image 的 fusion 配置要求
`data.dataset.image_size: [224, 224]`；radar-only 和包含 radar 的 fusion 配置要求 RA/DA
输入为 `128x64`，即默认 `clipped_range: 128` 和 `fft_tuple` 的第一/第三项为 `64/128`。
这些限制来自 motion mask、image/fusion teacher FC 输入和 radar branch 结构。

### 吞吐 profiling 与运行参数

训练前可以先用轻量 profile 脚本拆分 dataset、DataLoader、CPU 到 GPU 传输和 step 耗时：

```bash
conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/fusion/image_radar_gps_lidar_student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/fusion_io.json \
  --csv-output outputs/profile/fusion_io.csv
```

更多 image/radar/GPS/LiDAR/fusion profile 示例和 cache 复用规则见
`docs/training_throughput.md`。

包含 image 的实验建议先预热 image motion cache；包含 LiDAR 的实验建议先预热 LiDAR BEV cache。
并行运行多个实验时，默认配置使用较保守的 `num_workers: 4` 和 `prefetch_factor: 1`，避免多个
训练进程把 CPU worker 和预取队列成倍放大。单个实验可以通过命令行覆盖逐步调高：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_rkd.yaml \
  -o data.dataloader.num_workers=8 \
  -o data.dataloader.prefetch_factor=2
```

如果 DataLoader 使用 `pin_memory: true`，默认会启用 `training.transfer.non_blocking: true`，让
batch tensor 传输使用 non-blocking `.to(device)`。AMP 默认关闭；确认 cache 和 DataLoader 不再卡住后，
可以在 CUDA 上显式启用：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_rkd.yaml \
  -o training.amp.enabled=true \
  -o training.amp.dtype=float16
```

输出会写入 `outputs/<run_name>/`，包括：

- `final_config.yaml`
- `checkpoints/last.pth`
- `checkpoints/best.pth`
- `checkpoints/best_top1.pth`
- `metrics.json`
- `train_log.json`
- `training_outputs.npz`
- `artifacts/gps_scaler.npz`、`artifacts/lidar_normalizer.npz` 或 `artifacts/mmwave_scaler.npz`，仅在对应归一化启用时写入
- 训练曲线
- `tensorboard/` TensorBoard event 日志

恢复训练时设置 `training.resume=true` 会从当前 `output.run_name/checkpoints/last.pth` 恢复；也可以将
`training.resume` 设为 checkpoint 文件路径。恢复会加载模型、optimizer、scheduler、epoch 和最佳验证损失。

可以用 TensorBoard 查看和对比训练曲线：

```bash
tensorboard --logdir outputs
```

TensorBoard 标量包含基础训练曲线和验证平均指标：

- `accuracy/val_atop3`：所有 `J + 1` 个目标时隙 Top-3 accuracy 的平均值。
- `accuracy/val_atop5`：所有 `J + 1` 个目标时隙 Top-5 accuracy 的平均值。
- `dba/val_adba`：所有 `J + 1` 个目标时隙 DBA 的平均值，DBA 使用 Top-3 预测 beam 计算。

## 评估

```bash
python scripts/evaluate.py --config configs/image/teacher_no_kd.yaml --weights outputs/image_teacher_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/image/student_no_kd.yaml --weights outputs/image_student_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/radar/teacher_no_kd.yaml --weights outputs/radar_teacher_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/radar/student_no_kd.yaml --weights outputs/radar_student_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/gps/teacher_no_kd.yaml --weights outputs/gps_teacher_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/lidar/teacher_no_kd.yaml --weights outputs/lidar_teacher_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/mmwave/teacher_no_kd.yaml --weights outputs/mmwave_teacher_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/fusion/image_radar_rkd.yaml --weights outputs/image_radar_rkd/checkpoints/best.pth
```

评估会将指标和 `test_report.json` 写入配置的输出目录。未传 `--weights` 时，评估会尝试从
registry 中加载与当前配置匹配的最高验证 Top-1 checkpoint；如果 sidecar 记录了 GPS scaler、
LiDAR normalizer 或 mmWave scaler，评估会直接复用训练时工件，不再为了重新 fit 归一化状态扫描 train split。
如果 registry 缺失匹配项，则继续使用配置中的旧式权重路径回退。

## 预处理

```bash
python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml
python scripts/preprocess.py --config configs/preprocess/radar_da.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra_lidar.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps_lidar.yaml
python scripts/preprocess.py --config configs/preprocess/image_motion_cache.yaml
python scripts/preprocess.py --config configs/preprocess/lidar_bev_cache.yaml
```

所有单模态和 fusion 实验默认使用同一组包含 camera、radar、GPS、LiDAR 和可选 mmWave 列的序列 CSV：
`train_seqs_RA_GPS_LIDAR.csv` / `test_seqs_RA_GPS_LIDAR.csv`。运行
`configs/preprocess/sequences_ra_gps_lidar.yaml` 可生成这组统一 split；该配置会写出
`mmwave1..mmwave8`，默认来源列为 `unit1_pwr_60ghz`。GPS/mmWave scaler 只在训练集上 fit，并复用于测试集。

`configs/preprocess/image_motion_cache.yaml` 可把相邻 camera 帧 pair 的 motion mask 提前缓存为
`.npy` `uint8` 文件；训练配置中将 `image_motion_cache_dir` 指向该目录并启用
`image_motion_use_cache` 后可复用缓存。`configs/preprocess/lidar_bev_cache.yaml` 可把点云提前转换为
`.npy` BEV 缓存；训练配置中将 `lidar_cache_dir` 指向该目录并启用 `lidar_use_cache` 后可复用缓存。
BEV cache 会按 BEV 尺寸、ROI、FoV、ground/background 过滤参数自动分区，避免参数变化后误用旧缓存。
image motion cache 同样会按 image size、Gaussian sigma、阈值策略、灰度化方式和 cache version 自动分区。
这类原始模态预处理 cache 可以长期保留；训练 epoch、lr、batch size、num_workers、seed、模型结构和 KD
类型变化不会使它失效。原始 jpg/LiDAR/radar/GPS/beam 文件内容变化，或对应预处理参数变化时，应使用新的
参数 hash 目录或清理旧 cache。BEV 和 image cache 只会在读取当前样本时按需命中，不会在 dataset 初始化时
全量载入 cache 目录。

训练日志、评估报告和 `final_config.yaml` 会记录实际 split 路径和样本数，用于确认不同实验确实在
同一训练/测试集合上比较。

## 破坏性变更

旧的顶层入口脚本已移除。请改用以下新命令：

| 旧命令 | 新命令 |
| --- | --- |
| `python train_image.py ...` | `python scripts/train.py --config configs/image/<mode>.yaml ...` |
| `python train_both.py ...` | `python scripts/train.py --config configs/fusion/<mode>.yaml ...` |
| `python test_model_image.py ...` | `python scripts/evaluate.py --config configs/image/<mode>.yaml --weights <path>` |
| `python test_model_both.py ...` | `python scripts/evaluate.py --config configs/fusion/<mode>.yaml --weights <path>` |
| `python CSV_process.py ...` | `python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml` |
| `python gen_data_seq.py ...` | `python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml` |

## 组件

内置注册表位于 `kd_sensing.registries`：

- `MODELS`
- `DATASETS`
- `LOSSES`
- `METRICS`
- `DISTILLERS`
- `PREPROCESSORS`

关于如何添加新组件，请参见 [docs/extension_guide.md](docs/extension_guide.md)。
