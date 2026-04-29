我按 `openspec-explore` 只做了读取和验证，没有改代码。

**当前状态**
OpenSpec 没有活跃 change。`openspec validate --all` 通过，12 个 spec 全部 pass。  
在 README 指定的 `kd_mm_beam` 环境里，`pytest -q -p no:cacheprovider` 结果是 `187 passed`。实际 Scenario 9 数据首个样本的 image/radar/GPS/LiDAR 路径也能读通。

**主要问题**
1. OpenSpec 规格有明显过期内容。比如 GPS/Radar/LiDAR spec 仍写默认 `gru_params: [64, 64, 2]`，但 README、测试和当前配置已经统一为单模态 `[64, 64, 1]`。见 [gps-modality-model/spec.md](/root/projects/KD-for-sensing/openspec/specs/gps-modality-model/spec.md:23)、[radar-student-model/spec.md](/root/projects/KD-for-sensing/openspec/specs/radar-student-model/spec.md:43)、[lidar-modality-model/spec.md](/root/projects/KD-for-sensing/openspec/specs/lidar-modality-model/spec.md:86)、对照 [README.md](/root/projects/KD-for-sensing/README.md:101)。这会让后续按 spec 做变更时重新引入错误。

2. 序列生成有 off-by-one 风险。[sequences.py](/root/projects/KD-for-sensing/src/kd_sensing/preprocessing/sequences.py:63) 使用 `<`，会少取每个 `seq_index` 的最后一个合法窗口。按当前 Scenario 9、`in_len=8/out_len=3` 算，理论窗口 `4604`，当前生成逻辑是 `4468`，少了 `136` 条，正好每个序列少一条。

3. 小比例实验采样有偏。[samples.py](/root/projects/KD-for-sensing/src/kd_sensing/data/samples.py:29) 用 `head(int(len * portion))`，不是随机/分层采样。当前统一 split 里 `portion=0.05` 的 train 只覆盖 `seq_index 1..7`，test 只覆盖 `109..110`，用于快速 smoke 可以，但不能当代表性实验。

4. Teacher checkpoint 工作流比较脆。训练输出目录存在时会自动加时间戳，见 [trainer.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/trainer.py:48)；但 KD 配置解析 teacher 权重时固定找 `outputs/<slug>_teacher_no_kd/checkpoints/best.pth`，见 [builders.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/builders.py:240)。当前本地有 26 个 KD 配置默认 teacher 权重路径不存在，部分已有带时间戳 teacher 输出但不会被自动使用。

用户：能否建立一个文件夹用于存放所有最高精度checkpoints，checkpoints最好带上精度和模态，并修改代码优先从该文件夹加载对应 checkpoint？这样就算输出目录有时间戳也不会影响评估。并且是每次训练完成后，与当前配置精度最高的 checkpoint 都会被复制到该文件夹，命名格式可以是 `<slug>_teacher_no_kd_acc_<val_top1精度>.pth`。

5. GPS/LiDAR 归一化状态没有作为模型工件持久化。评估时会重新构建 train dataset 来拿 GPS scaler / LiDAR normalizer，见 [evaluator.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/evaluator.py:30)。只要训练 CSV、portion 或预处理结果变了，同一个 checkpoint 的评估输入标准化就可能变化。

6. `gps_smooth_window` 是死配置。参数一路传到 [transforms.py](/root/projects/KD-for-sensing/src/kd_sensing/data/transforms.py:539)，但 `build_gps_features` 实际没有使用它。这会误导使用者以为 GPS 平滑生效。

下一步最值得先立一个 OpenSpec change：统一修正 spec 过期内容，同时把序列 off-by-one、portion 采样语义、teacher checkpoint 解析策略纳入任务。