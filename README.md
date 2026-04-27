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
  radar/          # 仅雷达模型：RadarTeacher 无 KD 基线配置
  fusion/         # 图像+雷达模型：无 KD、logits KD、RKD 配置
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

python scripts/train.py --config configs/fusion/no_kd.yaml
python scripts/train.py --config configs/fusion/logits_kd.yaml
python scripts/train.py --config configs/fusion/rkd.yaml
```

这些配置复现 `All_models/*Std*.pth` 时会使用轻量 student 架构：image-only student 为
`image_student`，image+radar student 为 `fusion_student`。teacher 仍分别使用
`image_teacher` 和 `fusion_teacher`。上游旧训练脚本中 teacher-as-student 的实例化残留
不作为本项目配置驱动流程的语义依据。

Radar-only 基线用于复现论文表格中的 Radar 对照项。`configs/radar/no_kd.yaml`
将训练主模型 `model.student.type` 设置为 `radar_teacher`，表示在无 KD 模式下直接训练
RadarTeacher 架构；该配置不依赖仓库未提供的预训练 RadarTeacher 权重。

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
python scripts/evaluate.py --config configs/image/no_kd.yaml --weights All_models/ImageTeacher_noKD.pth
python scripts/evaluate.py --config configs/radar/no_kd.yaml --weights outputs/radar_no_kd/checkpoints/best.pth
python scripts/evaluate.py --config configs/fusion/rkd.yaml --weights All_models/BothStd_RKD.pth
```

评估会将指标和 `test_report.json` 写入配置的输出目录。

## 预处理

```bash
python scripts/preprocess.py --config configs/preprocess/radar_ra.yaml
python scripts/preprocess.py --config configs/preprocess/radar_da.yaml
python scripts/preprocess.py --config configs/preprocess/sequences_ra.yaml
```

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
