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
  fusion/         # 可选 image/radar/gps 融合模型配置
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

```bash
python scripts/train.py --config configs/image/no_kd.yaml
python scripts/train.py --config configs/image/logits_kd.yaml
python scripts/train.py --config configs/image/rkd.yaml

python scripts/train.py --config configs/radar/no_kd.yaml
python scripts/train.py --config configs/radar/student_no_kd.yaml
python scripts/train.py --config configs/radar/logits_kd.yaml
python scripts/train.py --config configs/radar/rkd.yaml

python scripts/train.py --config configs/fusion/no_kd.yaml
python scripts/train.py --config configs/fusion/logits_kd.yaml
python scripts/train.py --config configs/fusion/rkd.yaml

python scripts/train.py --config configs/gps/no_kd.yaml
python scripts/train.py --config configs/gps/student_no_kd.yaml
python scripts/train.py --config configs/gps/logits_kd.yaml
python scripts/train.py --config configs/gps/rkd.yaml
```

默认配置统一使用 `gru_params: [64, 64, 2]`，即所有内置单模态和多模态
teacher/student 都使用 2 层 GRU。仓库中随附的 `All_models/*Std*.pth` 和部分
`ImageTeacher*.pth` 是旧的一层 GRU 历史权重，不能作为当前默认配置的严格兼容权重；
按当前配置复现实验时需要重新训练或提供对应二层 GRU checkpoint。上游旧训练脚本中
teacher-as-student 的实例化残留不作为本项目配置驱动流程的语义依据。

Radar-only 配置中，`configs/radar/no_kd.yaml` 是 RadarTeacher baseline：
不加载 teacher，只用任务 loss 直接训练 `radar_teacher` 主模型，并作为 radar KD 的默认
teacher checkpoint 来源。`configs/radar/student_no_kd.yaml` 直接训练轻量
`radar_student`，用于对比无蒸馏的小模型表现。`logits_kd` 和 `rkd` 默认加载冻结的
`radar_teacher`，分别用 temperature soft logits KL loss 和关系蒸馏 loss 训练可训练的
`radar_student`。
配置注册名保持 `radar_teacher` 和 `radar_student`；对应 Python 类名分别为
`RadarModalityNet` 和 `RadarStudentModalityNet`，与 image/GPS 的 `*ModalityNet`
命名风格一致。
当前仓库没有内置 `All_models/RadarTeacher*.pth`，因此 radar KD 配置默认读取先运行
`configs/radar/no_kd.yaml` 生成的 `outputs/radar_no_kd/checkpoints/best.pth`。如果使用自定义
RadarTeacher 权重，可以覆盖路径：

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

Fusion 模型通过 `model.teacher.modalities` 和 `model.student.modalities` 选择参与融合的模态，
可用值为 `image`、`radar`、`gps`。旧配置显式保留 `modalities: [image, radar]`；新增示例：

```bash
python scripts/train.py --config configs/fusion/image_gps_no_kd.yaml
python scripts/train.py --config configs/fusion/radar_gps_no_kd.yaml
python scripts/train.py --config configs/fusion/all_modalities_no_kd.yaml
```

可以使用点号分隔的键覆盖配置值：

```bash
python scripts/train.py --config configs/image/rkd.yaml training.epochs=1 data.dataset.portion=0.05
```

输出会写入 `outputs/<run_name>/`，包括：

- `final_config.yaml`
- `checkpoints/last.pth`
- `checkpoints/best.pth`
- `metrics.json`
- `train_log.json`
- `training_outputs.npz`
- 训练曲线
- `tensorboard/` TensorBoard event 日志

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
python scripts/evaluate.py --config configs/image/no_kd.yaml --weights outputs/image_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/radar/no_kd.yaml --weights outputs/radar_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/radar/student_no_kd.yaml --weights outputs/radar_student_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/gps/no_kd.yaml --weights outputs/gps_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/fusion/rkd.yaml --weights outputs/fusion_rkd/checkpoints/best.pth
```

评估会将指标和 `test_report.json` 写入配置的输出目录。

## 预处理

```bash
python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml
python scripts/preprocess.py --config configs/preprocess/radar_da.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps.yaml
```

GPS 实验需要带 `gps1..gps8` 和 `bs_gps1..bs_gps8` 列的序列 CSV。运行
`configs/preprocess/sequences_ra_gps.yaml` 后会生成 `train_seqs_RA_GPS.csv` 和
`test_seqs_RA_GPS.csv`，供 GPS-only 和启用 GPS 的 fusion 配置使用。GPS scaler 只在训练集
上 fit，并复用于测试集。

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
