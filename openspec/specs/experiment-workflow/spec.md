# experiment-workflow Specification

## Purpose
TBD - created by archiving change reorganize-project-structure. Update Purpose after archive.
## Requirements
### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、teacher/student 模型、KD 模式、训练超参数、优化器、调度器、输出目录、随机种子、GPS-Rel-Polar 特征模式和 fusion 模态选择。

#### Scenario: 使用配置启动 image-only 训练
- **WHEN** 用户通过新 CLI 传入 image-only 训练配置
- **THEN** 系统 MUST 构建 image-only dataset、teacher/student 模型、KD/loss、optimizer 和 scheduler，并进入训练流程

#### Scenario: 使用配置启动 image+radar 训练
- **WHEN** 用户通过新 CLI 传入 fusion 训练配置
- **THEN** 系统 MUST 构建同时包含图像和雷达输入的 dataset、fusion teacher/student 模型、KD/loss、optimizer 和 scheduler，并进入训练流程

#### Scenario: 使用配置启动 radar-only 训练
- **WHEN** 用户通过新 CLI 传入 radar-only 训练配置
- **THEN** 系统 MUST 构建包含雷达输入的 dataset、配置指定的 radar-only 主模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 支持 `radar_teacher` baseline 和 `radar_student` lightweight student
- **AND** 训练流程 MUST 不要求模型接收图像输入

#### Scenario: 使用配置启动 radar-only 评估
- **WHEN** 用户通过新 CLI 传入 radar-only 评估配置和 radar-only 模型权重
- **THEN** 系统 MUST 构建配置指定的 radar-only 模型并只使用雷达输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

#### Scenario: 使用配置启动 GPS-only 训练
- **WHEN** 用户通过新 CLI 传入 GPS-only 训练配置
- **THEN** 系统 MUST 构建包含 GPS 输入的 dataset、配置指定的 GPS teacher/student 模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求模型接收图像或雷达输入
- **AND** GPS 输入 MUST 使用 `relative_polar` 三维特征

#### Scenario: 使用配置启动 GPS-only 评估
- **WHEN** 用户通过新 CLI 传入 GPS-only 评估配置和 GPS 模型权重
- **THEN** 系统 MUST 构建配置指定的 GPS 模型并只使用 GPS 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标
- **AND** GPS 输入 MUST 使用 `relative_polar` 三维特征

#### Scenario: 使用配置启动可选模态 fusion 训练
- **WHEN** 用户通过新 CLI 传入带 `modalities` 的 fusion 配置
- **THEN** 系统 MUST 只准备并融合 `modalities` 中列出的模态
- **AND** 系统 MUST 支持 image、radar、gps 的任意非空组合

### Requirement: 命令行覆盖配置
实验入口 MUST 支持在命令行覆盖配置值。新 CLI MUST 支持显式传入配置文件和关键参数覆盖；旧脚本 argparse 参数不得作为兼容入口保留，只能作为迁移默认值参考。

#### Scenario: 覆盖训练轮数
- **WHEN** 用户通过命令行将训练轮数覆盖为 `1`
- **THEN** 系统 MUST 使用覆盖后的训练轮数，而不是配置文件中的默认训练轮数

#### Scenario: 覆盖 KD 模式
- **WHEN** 用户通过命令行将 `kd_mode` 覆盖为 no-KD、logits KD 或 RKD 中的一种
- **THEN** 系统 MUST 构建对应蒸馏逻辑，并保持该模式下原有损失计算语义

### Requirement: 统一实验输出
训练和评估流程 MUST 将运行产物写入统一输出目录。输出目录 MUST 至少包含本次运行的有效配置、checkpoint 或权重引用、metrics、训练曲线或日志，以及测试报告。训练和评估流程 MUST 默认创建互不覆盖的运行目录；只有在用户显式启用覆盖、显式恢复训练或传入确定性输出目录时，系统才 MAY 复用既有目录。训练流程 MUST 在启用 TensorBoard 时写入可由 TensorBoard 读取的标量 event 日志，并且 MUST 支持通过配置关闭该日志写入。训练流程 MUST 在启用进度显示时提供 `tqdm` 训练进度条，并且 MUST 将每个 epoch 的进度摘要保存到运行日志。

#### Scenario: 训练完成后保存运行配置
- **WHEN** 一次训练任务启动并创建输出目录
- **THEN** 系统 MUST 保存解析和覆盖后的最终配置，便于后续复现实验

#### Scenario: 固定 run_name 默认不覆盖旧实验
- **WHEN** 用户设置 `output.run_name` 且目标运行目录已存在，并且未启用覆盖或 resume
- **THEN** 系统 MUST 创建带唯一 run id、时间戳或等价后缀的新运行目录
- **AND** 系统 MUST 不覆盖既有 `final_config.yaml`、`metrics.json`、checkpoint、TensorBoard event 或 `train_log.json`

#### Scenario: 显式恢复训练复用运行目录
- **WHEN** 用户设置 `training.resume: true` 且提供固定 `output.run_name`
- **THEN** 系统 MUST 从该运行目录下的 checkpoint 恢复训练
- **AND** 系统 MUST 不自动追加新的 run id

#### Scenario: 显式覆盖运行目录
- **WHEN** 用户显式启用输出覆盖配置并设置 `output.run_name`
- **THEN** 系统 MAY 复用该运行目录
- **AND** 系统 MUST 在最终配置或运行日志中记录覆盖行为

#### Scenario: 训练过程中显示 tqdm 进度
- **WHEN** 一次训练任务启动且进度显示配置启用
- **THEN** 系统 MUST 使用 `tqdm` 展示 epoch 或 batch 级训练进度
- **AND** 进度条 MUST 展示当前 epoch、batch 进度、训练损失、任务损失、蒸馏损失、训练准确率和学习率中的关键状态

#### Scenario: 训练完成后保存进度日志
- **WHEN** 一次训练任务完成至少一个 epoch
- **THEN** 系统 MUST 在当前运行目录的训练日志中保存 epoch 级进度摘要
- **AND** 进度摘要 MUST 包含 epoch 编号、训练损失、训练任务损失、训练蒸馏损失、训练准确率、验证损失、验证准确率和学习率
- **AND** 日志保存 MUST 保持既有历史指标数组兼容

#### Scenario: 通过配置关闭 tqdm 进度显示
- **WHEN** 用户在训练配置中关闭进度显示
- **THEN** 系统 MUST 不创建可视化 `tqdm` 进度条
- **AND** 系统 MUST 继续保存训练日志和 epoch 级进度摘要

#### Scenario: 训练过程中写入 TensorBoard 标量日志
- **WHEN** 一次训练任务完成至少一个 epoch 且 TensorBoard 日志启用
- **THEN** 系统 MUST 在当前运行目录下写入 TensorBoard event 文件
- **AND** event 文件 MUST 记录训练总损失、训练任务损失、训练蒸馏损失、训练准确率、验证损失、验证准确率、学习率、验证 `ATop-3`、验证 `ATop-5` 和验证 `ADBA` 标量

#### Scenario: TensorBoard 记录跨时隙平均验证指标
- **WHEN** 一次训练 epoch 的验证阶段产出 per-slot Top-K accuracy 和 DBA 结果
- **THEN** 系统 MUST 将 `ATop-3` 计算为所有 `J + 1` 个目标时隙 Top-3 accuracy 的平均值
- **AND** 系统 MUST 将 `ATop-5` 计算为所有 `J + 1` 个目标时隙 Top-5 accuracy 的平均值
- **AND** 系统 MUST 将 `ADBA` 计算为所有 `J + 1` 个目标时隙 DBA 的平均值，且 DBA MUST 使用 Top-3 预测 beam 计算

#### Scenario: 通过配置关闭 TensorBoard 日志
- **WHEN** 用户在训练配置中关闭 TensorBoard 日志
- **THEN** 系统 MUST 不创建新的 TensorBoard writer，也不得因未写入 event 文件而影响训练完成

#### Scenario: 评估完成后保存指标
- **WHEN** 一次评估任务完成
- **THEN** 系统 MUST 在输出目录保存 Top-K、DBA、loss、latency 或当前评估入口支持的指标结果

#### Scenario: 评估默认不覆盖旧报告
- **WHEN** 用户多次运行评估入口且未显式指定覆盖同一输出目录
- **THEN** 系统 MUST 为每次评估创建互不覆盖的评估运行目录
- **AND** 系统 MUST 不固定覆盖 `outputs/evaluation/test_report.json`

#### Scenario: 输出记录 split 和样本数
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** 系统 MUST 在最终配置、运行日志或测试报告中记录实际使用的 train/test CSV 路径
- **AND** 系统 MUST 记录每个 split 的样本数，便于判断不同实验是否可横向比较

### Requirement: 跨模态可比较 split 配置
项目 MUST 提供可用于单模态和多模态横向比较的统一 split 配置方式。默认和 canonical 实验配置 MUST 让 image、radar、GPS、LiDAR、mmWave 和 fusion 实验引用同一组 train/test CSV。默认统一 CSV 文件名 MUST 继续可配置为 `train_seqs_RA_GPS_LIDAR.csv` 和 `test_seqs_RA_GPS_LIDAR.csv`；当启用 mmWave 时，这组 CSV MUST 由预处理流程生成可选的 `mmwave1..mmwaveN` 列。

#### Scenario: 使用统一 split 运行单模态实验
- **WHEN** 用户将 image、radar、GPS、LiDAR 和 mmWave 单模态配置指向同一组 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建各模态 dataset
- **AND** 训练或评估输出 MUST 记录相同的 CSV 路径和样本数

#### Scenario: 使用统一 split 运行 fusion 实验
- **WHEN** 用户将 fusion 配置指向与单模态相同的 train/test CSV
- **THEN** 系统 MUST 使用相同样本集合构建 fusion dataset
- **AND** 未启用模态不得影响该 split 的可用性

#### Scenario: 默认配置使用统一 split
- **WHEN** 开发者查看默认 image、radar、GPS、LiDAR、mmWave 和 fusion 实验配置
- **THEN** 每个配置 MUST 指向同一组 train/test CSV，默认文件名为 `train_seqs_RA_GPS_LIDAR.csv` 和 `test_seqs_RA_GPS_LIDAR.csv`
- **AND** 输出 MUST 清晰记录该统一 split 的路径和样本数

#### Scenario: mmWave 统一 split 包含输入列
- **WHEN** 用户运行默认 mmWave 或包含 mmWave 的 fusion 配置
- **THEN** 该配置引用的统一 split CSV MUST 包含 `mmwave1..mmwaveN` 列
- **AND** 如果列缺失，系统 MUST 抛出清晰错误并提示重新运行启用 mmWave 的序列预处理

### Requirement: 训练与评估行为等价
结构重构后，默认 image-only、radar-only、GPS-only、LiDAR-only 和 fusion 工作流 MUST 通过新脚本保持当前算法的核心训练、验证和评估语义，包括默认序列长度、预测步数、类别数、KD 模式、teacher 权重选择、student 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。上游原代码实际覆盖的 image-only 与 image+radar 配置 MUST 按原代码和随附参数文件对齐 GRU 层数与训练超参数；radar-only、GPS-only 和 LiDAR-only 是本项目新增单模态配置，MUST 在共享字段上与 image 单模态配置保持一致。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认任务语义，并保持相同的任务类型
- **AND** `configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 中的单模态 teacher 与 student `gru_params` MUST 为 `[64, 64, 1]`
- **AND** `configs/radar/*.yaml`、`configs/gps/*.yaml` 和 `configs/lidar/*.yaml` 中的共享训练字段 MUST 与 `configs/image/` 下同角色配置一致
- **AND** `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml` 中的 image+radar fusion teacher `gru_params` MUST 为 `[64, 64, 2]`
- **AND** `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml`、`configs/fusion/rkd.yaml` 和 `configs/fusion/image_radar_*.yaml` 中的 image+radar fusion student `gru_params` MUST 为 `[64, 64, 1]`
- **AND** image+radar teacher no-KD 配置中作为训练主模型的 `model.student` 若为 `fusion_teacher`，其 `gru_params` MUST 为 `[64, 64, 2]`
- **AND** `src/kd_sensing/config/defaults.py` MUST 不把所有 teacher/student 的 `gru_params` 统一强制为 `[64, 64, 2]`

#### Scenario: 默认 student 架构与 GRU 层数
- **WHEN** 用户使用默认 image-only、radar-only、GPS-only、LiDAR-only 或 fusion student 实验配置构建模型
- **THEN** 系统 MUST 为 image-only 工作流构建轻量 `image_student`
- **AND** 系统 MUST 为 radar-only 工作流构建轻量 `radar_student`
- **AND** 系统 MUST 为 GPS-only 工作流构建轻量 `gps_student`
- **AND** 系统 MUST 为 LiDAR-only 工作流构建轻量 `lidar_student`
- **AND** 系统 MUST 为 fusion 工作流构建轻量 `fusion_student`
- **AND** image、radar、GPS 和 LiDAR 单模态 student 模型的 `GRU.num_layers` MUST 为 1
- **AND** 原代码兼容 image+radar fusion student 模型的 `GRU.num_layers` MUST 为 1
- **AND** 文档 MUST 说明二层 GRU student 是历史 canonical 配置或特定扩展配置，不是当前单模态和 image+radar 兼容配置的默认结构

#### Scenario: 默认 teacher GRU 层数
- **WHEN** 用户通过目标兼容配置构建 image、radar、GPS、LiDAR 或 image+radar fusion teacher 模型
- **THEN** image、radar、GPS 和 LiDAR 单模态 teacher 模型的 `GRU.num_layers` MUST 为 1
- **AND** image、radar、GPS 和 LiDAR 单模态 teacher 配置 MUST 使用 `gru_params: [64, 64, 1]`
- **AND** image+radar fusion teacher 模型的 `GRU.num_layers` MUST 为 2
- **AND** image+radar fusion teacher 配置 MUST 使用 `gru_params: [64, 64, 2]`

#### Scenario: checkpoint 恢复语义
- **WHEN** 用户在训练配置中启用 `training.resume`
- **THEN** 训练流程 MUST 尝试恢复 checkpoint
- **AND** 恢复 MUST 包含模型权重、optimizer、scheduler、已完成 epoch 和 best validation loss
- **AND** `training.start_epoch` MUST 不再是唯一影响恢复 epoch 的字段

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
- **AND** 使用目标兼容配置时，smoke test MUST 使用与该配置匹配的 GRU 层数构建模型

### Requirement: Radar-only KD 实验配置
项目 MUST 提供 radar-only KD 配置，使 radar-only 实验能够通过配置选择 `logits_kd` 和 `rkd` 蒸馏模式。KD 配置 MUST 使用 `experiment.task: radar`，MUST 通过 `radar_teacher` 构建 frozen teacher，MUST 通过 `radar_student` 构建可训练 student，MUST 配置可解析的 RadarTeacher checkpoint 来源，并且 MUST 继续复用统一训练入口、loss、optimizer、scheduler、验证指标和输出目录语义。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/logits_kd.yaml`
- **THEN** 系统 MUST 构建 `logits_kd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_student` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练

#### Scenario: 使用 RKD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/rkd.yaml`
- **THEN** 系统 MUST 构建 `rkd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_student` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练

#### Scenario: 使用默认 RadarTeacher checkpoint
- **WHEN** 用户未覆盖 radar KD 配置中的 teacher 权重字段
- **THEN** 系统 MUST 从 radar teacher no-KD 训练输出目录解析 teacher checkpoint
- **AND** 该默认路径 MUST 对应 `outputs/radar_no_kd/checkpoints/best.pth`

#### Scenario: 覆盖 RadarTeacher checkpoint
- **WHEN** 用户通过命令行覆盖 `paths.weights_dir` 或 `distillation.teacher_model_name`
- **THEN** 系统 MUST 使用覆盖后的值解析 radar teacher checkpoint
- **AND** 系统 MUST 保持其它 radar-only KD 配置语义不变

### Requirement: RadarStudent no-KD 实验配置
项目 MUST 提供 radar-only lightweight student no-KD 配置，用于直接训练 `radar_student` 并评估轻量雷达模型在无蒸馏条件下的表现。该配置 MUST 使用 `experiment.task: radar`，MUST 不加载 teacher checkpoint，并 MUST 复用统一训练、验证、评估和输出目录语义。

#### Scenario: 使用 no-KD 启动 RadarStudent 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/student_no_kd.yaml`
- **THEN** 系统 MUST 构建 `radar_student` 作为可训练主模型
- **AND** 系统 MUST 不构建或加载 frozen teacher
- **AND** 系统 MUST 只使用雷达输入完成 forward

### Requirement: 预处理流程可单独运行
CSV 处理和序列生成 MUST 通过新预处理脚本或包内 CLI 作为独立入口提供，并支持配置指定输入 CSV、数据根目录、输出 CSV 名称、FFT 参数、处理比例和是否输出 GPS 序列列。

#### Scenario: 运行 CSV 预处理
- **WHEN** 用户通过新预处理入口指定 Scenario 9 原始 CSV 和数据根目录
- **THEN** 系统 MUST 生成符合当前数据格式的 RA/DA CSV 或中间文件引用

#### Scenario: 运行序列生成
- **WHEN** 用户通过新预处理入口指定已处理 CSV 和输出目录
- **THEN** 系统 MUST 生成训练和测试序列 CSV，供统一 dataset 配置引用

#### Scenario: 运行带 GPS 列的序列生成
- **WHEN** 用户通过新预处理入口启用 GPS 序列列输出
- **THEN** 系统 MUST 在训练和测试序列 CSV 中写入历史 GPS 路径列
- **AND** 输出 CSV MUST 可被启用 GPS 的 Scenario 9 dataset 直接读取

### Requirement: GPS-Rel-Polar 实验配置
项目 MUST 提供 GPS-only 的 GPS-Rel-Polar 配置，使用户能通过统一训练入口运行选定的 GPS 表示。每个 GPS 配置 MUST 明确 `gps_feature_mode: relative_polar`、`gps_input_size: 3`、模型注册名、输出目录和 run name。

#### Scenario: 运行 GPS-Rel-Polar 训练
- **WHEN** 用户通过训练入口运行 GPS-Rel-Polar 配置
- **THEN** 系统 MUST 使用 `gps_feature_mode: relative_polar`
- **AND** 系统 MUST 构建 `gps_teacher` 或 `gps_student` 的 `gps_input_size` 为 3

#### Scenario: 不提供其它 GPS ablation 配置
- **WHEN** 用户查看本 change 提供的 GPS 配置入口
- **THEN** 系统 MUST 不提供 raw、UTM、relative、motion 或 motion-smooth 的独立 GPS ablation 配置作为受支持入口
- **AND** GPS 配置文档 MUST 引导用户使用 GPS-Rel-Polar

### Requirement: 可选模态 fusion 实验配置
项目 MUST 提供可选模态 fusion 配置，使用户能通过 `modalities` 手动选择 `image`、`radar`、`gps` 的任意非空组合。Fusion KD 配置 MUST 要求 teacher 和 student 使用相同的 `modalities`，除非用户显式选择受支持的跨模态蒸馏配置。

#### Scenario: 运行 image+gps fusion
- **WHEN** 用户运行 `modalities: ["image", "gps"]` 的 fusion 配置
- **THEN** 系统 MUST 构建只包含 image 和 gps 分支的 fusion teacher/student
- **AND** 系统 MUST 不要求 radar 输入

#### Scenario: 运行 radar+gps fusion
- **WHEN** 用户运行 `modalities: ["radar", "gps"]` 的 fusion 配置
- **THEN** 系统 MUST 构建只包含 radar 和 gps 分支的 fusion teacher/student
- **AND** 系统 MUST 不要求 image 输入

#### Scenario: 运行 image+radar+gps fusion
- **WHEN** 用户运行 `modalities: ["image", "radar", "gps"]` 的 fusion 配置
- **THEN** 系统 MUST 构建包含全部三种模态分支的 fusion teacher/student
- **AND** 系统 MUST 使用统一训练、验证和评估流程输出指标

### Requirement: LiDAR 配置驱动实验
项目 MUST 支持通过配置文件启动 LiDAR-only 训练和评估。LiDAR-only 配置 MUST 使用 `experiment.task: lidar`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 LiDAR-only 训练
- **WHEN** 用户通过新 CLI 传入 LiDAR-only 训练配置
- **THEN** 系统 MUST 构建包含 LiDAR 输入的 dataset、配置指定的 LiDAR teacher/student 模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达或 GPS 输入
- **AND** LiDAR 输入 MUST 使用 BEV 张量格式

#### Scenario: 使用配置启动 LiDAR-only 评估
- **WHEN** 用户通过新 CLI 传入 LiDAR-only 评估配置和 LiDAR 模型权重
- **THEN** 系统 MUST 构建配置指定的 LiDAR 模型并只使用 LiDAR 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

### Requirement: LiDAR fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 LiDAR。包含 LiDAR 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程。

#### Scenario: 使用配置启动 image+radar+gps+lidar fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar"]` 的 fusion 配置
- **THEN** 系统 MUST 构建四个模态输入所需的 dataset 字段和 fusion teacher/student 模型
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps 和 lidar 输入

#### Scenario: 使用配置启动 LiDAR 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `lidar` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: LiDAR 默认实验配置
项目 MUST 提供 LiDAR-only no-KD、LiDAR student no-KD、LiDAR logits KD、LiDAR RKD 和包含 LiDAR 的 fusion 示例配置。所有默认 LiDAR teacher/student 配置 MUST 使用 `gru_params: [64, 64, 2]`。

#### Scenario: LiDAR 默认配置可构建
- **WHEN** 开发者加载 `configs/lidar/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、distiller、optimizer 和 scheduler
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: LiDAR fusion 示例配置可构建
- **WHEN** 开发者加载包含 LiDAR 的 `configs/fusion/*.yaml`
- **THEN** 系统 MUST 能构建对应 fusion teacher 和 fusion student
- **AND** fusion `modalities` MUST 只包含合法模态名称

### Requirement: LiDAR 预处理入口
预处理 CLI MUST 支持生成带 LiDAR 路径列的序列 CSV，并支持按配置离线生成 LiDAR BEV 缓存。

#### Scenario: 生成 LiDAR 序列 CSV
- **WHEN** 用户运行预处理入口并选择 sequence CSV 生成且启用 LiDAR
- **THEN** 系统 MUST 输出包含 `lidar1..lidarN` 的 train/test 序列 CSV

#### Scenario: 生成 LiDAR BEV 缓存
- **WHEN** 用户运行预处理入口并选择 LiDAR BEV 缓存生成
- **THEN** 系统 MUST 根据配置读取点云、应用裁剪和 BEV 构造，并写出可被 dataset 读取的 `.npy` 缓存

### Requirement: LiDAR dry-run 训练
项目 MUST 提供可在小数据或 fixture 上运行的 LiDAR smoke test 路径，用于验证 LiDAR forward、loss、backward、validation 和 checkpoint 保存。

#### Scenario: LiDAR dry-run 训练
- **WHEN** 开发者使用 synthetic、fixture 或小比例数据运行一次 LiDAR 短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径

### Requirement: 单模态 canonical 配置矩阵
项目 MUST 为每个受支持单模态 `image`、`radar`、`gps` 和 `lidar` 提供统一命名的 canonical 配置矩阵。每个单模态目录 MUST 包含 `teacher_no_kd.yaml`、`student_no_kd.yaml`、`logits_kd.yaml` 和 `rkd.yaml`。canonical 配置 MUST 使用统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和输出目录语义。

#### Scenario: 单模态 teacher no-KD 配置
- **WHEN** 开发者加载 `configs/<modality>/teacher_no_kd.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 设置 `distillation.teacher_model_name: null`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_teacher`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_teacher_no_kd`

#### Scenario: 单模态 student no-KD 配置
- **WHEN** 开发者加载 `configs/<modality>/student_no_kd.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 设置 `distillation.teacher_model_name: null`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_student`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_student_no_kd`

#### Scenario: 单模态 logits KD 配置
- **WHEN** 开发者加载 `configs/<modality>/logits_kd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: logits_kd`
- **AND** 配置 MUST 构建 frozen `<modality>_teacher`
- **AND** 配置 MUST 构建可训练 `<modality>_student`
- **AND** 配置 MUST 默认解析对应 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: 单模态 RKD 配置
- **WHEN** 开发者加载 `configs/<modality>/rkd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: rkd`
- **AND** 配置 MUST 构建 frozen `<modality>_teacher`
- **AND** 配置 MUST 构建可训练 `<modality>_student`
- **AND** 配置 MUST 提供 `rkd_pairs_per_anchor`、`rkd_distance_weight` 和 `rkd_angle_weight`
- **AND** 配置 MUST 默认解析对应 canonical teacher no-KD 输出中的 `best.pth`

### Requirement: 单模态 legacy no-KD 入口兼容
项目 MUST 保留现有 `configs/<modality>/no_kd.yaml` 入口作为兼容配置，并 MUST 在文档中说明其历史语义和推荐替代入口。legacy 入口不得改变 canonical 配置矩阵的语义。

#### Scenario: image legacy no-KD 保持 student baseline
- **WHEN** 用户运行 `configs/image/no_kd.yaml`
- **THEN** 系统 MUST 继续训练 `image_student`
- **AND** 文档 MUST 引导新实验优先使用 `configs/image/student_no_kd.yaml`

#### Scenario: radar GPS LiDAR legacy no-KD 保持 teacher baseline
- **WHEN** 用户运行 `configs/radar/no_kd.yaml`、`configs/gps/no_kd.yaml` 或 `configs/lidar/no_kd.yaml`
- **THEN** 系统 MUST 继续训练对应 teacher baseline
- **AND** 文档 MUST 引导新实验优先使用对应 `teacher_no_kd.yaml`

### Requirement: teacher/student 角色不得受原脚本残留影响
配置驱动流程 MUST 以 YAML 中的 `model.student` 作为 no-KD 时的被训练主模型，并 MUST 只在 `distillation.type` 非 `no_kd` 时构建 frozen teacher。默认 canonical student baseline 和 KD 配置 MUST 使用 lightweight student，不得默认使用 teacher-as-student 残留。

#### Scenario: no-KD 只训练配置中的主模型
- **WHEN** 配置设置 `distillation.type: no_kd`
- **THEN** 训练流程 MUST 不构建或加载 frozen teacher
- **AND** optimizer MUST 只更新 `model.student` 构建出的主模型

#### Scenario: canonical student baseline 使用 lightweight student
- **WHEN** 开发者加载任意 canonical `student_no_kd.yaml`
- **THEN** `model.student.type` MUST 为对应 lightweight student 注册名
- **AND** `model.student.type` MUST NOT 等于对应 teacher 注册名

#### Scenario: canonical KD 使用 teacher 蒸馏 student
- **WHEN** 开发者加载任意 canonical `logits_kd.yaml` 或 `rkd.yaml`
- **THEN** `model.teacher.type` MUST 为对应 teacher 注册名
- **AND** `model.student.type` MUST 为对应 lightweight student 注册名
- **AND** teacher 和 student 的输出 hidden size MUST 对齐以支持 RKD

### Requirement: canonical 配置命名与输出目录一致
canonical 配置 MUST 使用可预测的实验名、run name 和默认 teacher checkpoint 来源。默认路径 MUST 便于用户按 teacher baseline -> student baseline/KD 的顺序运行实验，并 MUST 支持命令行覆盖。

#### Scenario: canonical run name 与文件语义一致
- **WHEN** 开发者加载任意 canonical 配置
- **THEN** `experiment.name` MUST 与不含 `.yaml` 的文件 stem 一致
- **AND** `output.run_name` MUST 与 `experiment.name` 一致

#### Scenario: canonical KD 默认读取 teacher baseline 输出
- **WHEN** 用户未覆盖 canonical KD 配置中的 teacher 权重字段
- **THEN** 系统 MUST 从对应 canonical `teacher_no_kd` 输出目录解析 teacher checkpoint
- **AND** 默认 checkpoint 文件名 MUST 为 `best.pth`

#### Scenario: canonical KD checkpoint 可覆盖
- **WHEN** 用户通过命令行覆盖 `paths.weights_dir` 或 `distillation.teacher_model_name`
- **THEN** 系统 MUST 使用覆盖后的 teacher checkpoint 来源
- **AND** 系统 MUST 保持该配置的 teacher/student 模型角色不变

### Requirement: 稳定实验工件输出记录
训练和评估流程 MUST 在最终配置、训练日志或测试报告中记录 checkpoint 解析与归档信息。记录内容 MUST 包含实际加载 checkpoint 路径、加载来源、registry 目录、归档 checkpoint 路径、验证 Top-1 accuracy、归一化工件路径和实际 split 样本数。

#### Scenario: 训练日志记录归档结果
- **WHEN** 一次训练完成并启用最佳 checkpoint 归档
- **THEN** `train_log.json` 或等价训练日志 MUST 记录 registry 目录和归档 checkpoint 路径
- **AND** 日志 MUST 记录用于归档命名的验证 Top-1 accuracy
- **AND** 日志 MUST 继续记录 train/test CSV 路径和样本数

#### Scenario: 评估报告记录权重来源
- **WHEN** 一次评估加载 checkpoint
- **THEN** `test_report.json` MUST 记录最终 checkpoint 路径
- **AND** 报告 MUST 记录 checkpoint 来源是显式路径、registry 还是旧路径回退

### Requirement: 默认实验 checkpoint 可被时间戳输出目录解耦
默认 KD 和评估工作流 MUST 不依赖固定 `outputs/<run_name>/checkpoints/best.pth` 作为唯一权重来源。当固定 `run_name` 已存在导致新训练输出目录追加时间戳时，后续 KD 或评估 MUST 能通过 registry 找到对应配置的最高验证 Top-1 checkpoint。

#### Scenario: 时间戳 teacher 输出被 KD 复用
- **WHEN** teacher no-KD 训练因为目标运行目录已存在而写入带时间戳后缀的新运行目录
- **THEN** 训练完成后 registry MUST 保存该 teacher 的最高验证 Top-1 checkpoint
- **AND** 后续对应 KD 配置 MUST 能从 registry 加载该 teacher checkpoint

#### Scenario: 旧路径保持兼容
- **WHEN** 用户已有旧式 `paths.weights_dir / teacher_model_name` checkpoint 且 registry 没有匹配候选
- **THEN** KD teacher 加载 MUST 继续支持旧路径

### Requirement: mmWave 配置驱动实验
项目 MUST 支持通过配置文件启动 mmWave-only 训练和评估。mmWave-only 配置 MUST 使用 `experiment.task: mmwave`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 mmWave-only 训练
- **WHEN** 用户通过新 CLI 传入 mmWave-only 训练配置
- **THEN** 系统 MUST 构建包含 mmWave 输入的 dataset、配置指定的 mmWave teacher/student 模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达、GPS 或 LiDAR 输入
- **AND** mmWave 输入 MUST 使用 `[B, T, 64]` 的 dB receive-power 特征序列

#### Scenario: 使用配置启动 mmWave-only 评估
- **WHEN** 用户通过新 CLI 传入 mmWave-only 评估配置和 mmWave 模型权重
- **THEN** 系统 MUST 构建配置指定的 mmWave 模型并只使用 mmWave 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标
- **AND** 评估流程 MUST 复用训练时保存的 mmWave scaler

### Requirement: mmWave fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 mmWave。包含 mmWave 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程。

#### Scenario: 使用配置启动五模态 fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]` 的 fusion 配置
- **THEN** 系统 MUST 构建五个模态输入所需的 dataset 字段和 fusion teacher/student 模型
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps、lidar 和 mmWave 输入

#### Scenario: 使用配置启动 mmWave 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `mmwave` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: mmWave 默认实验配置
项目 MUST 提供 mmWave-only teacher no-KD、student no-KD、logits KD、RKD 配置和包含 mmWave 的 canonical fusion 配置。所有默认 mmWave teacher/student 配置 MUST 使用 `mmwave_input_size: 64`、`mmwave_normalize: true` 和 `gru_params: [64, 64, 1]`。

#### Scenario: mmWave 默认配置可构建
- **WHEN** 开发者加载 `configs/mmwave/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、distiller、optimizer 和 scheduler
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** teacher 和 student 配置的 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave KD 配置默认 checkpoint 来源
- **WHEN** 用户运行 `configs/mmwave/logits_kd.yaml` 或 `configs/mmwave/rkd.yaml` 且未显式覆盖 teacher 权重
- **THEN** 系统 MUST 从 mmWave teacher no-KD 训练输出或最佳 checkpoint registry 解析 teacher checkpoint
- **AND** 该默认解析 MUST 与其它单模态 KD 配置的 checkpoint 优先级一致

### Requirement: mmWave 预处理入口
预处理入口 MUST 支持通过配置生成带 mmWave 输入列的 Scenario 9 sequence CSV。该入口 MUST 允许配置 mmWave 源列和 fallback 列，并保持未启用 mmWave 的序列生成行为兼容。

#### Scenario: 运行带 mmWave 列的序列生成
- **WHEN** 用户通过预处理入口启用 `include_mmwave: true`
- **THEN** 系统 MUST 在训练和测试序列 CSV 中写入历史 `mmwave1..mmwaveN` 路径列
- **AND** 输出 CSV MUST 可被启用 mmWave 的 Scenario 9 dataset 直接读取

#### Scenario: mmWave 源列缺失
- **WHEN** 用户启用 mmWave 序列列输出但原始 CSV 不包含配置的 mmWave 源列或 fallback 列
- **THEN** 系统 MUST 抛出包含源列名、fallback 列名和 CSV 路径的清晰错误

### Requirement: 实验入口自动解析 cache policy
训练、评估和 profile 入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的实际 cache 读写开关传递给 dataset。解析过程 MUST 使用配置中的启用模态，不得要求用户为每个单模态或 fusion 组合手动设置低层 cache 读写字段。

#### Scenario: 单模态训练自动解析
- **WHEN** 用户运行 `configs/image/teacher_no_kd.yaml` 且未手动设置 image cache 低层开关
- **THEN** 训练入口 MUST 根据 cache policy 自动决定 `image_motion_use_cache` 和 `image_motion_write_cache`
- **AND** 用户 MUST 能通过命令行覆盖这些低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 训练入口 MUST 只为该组合包含的 image 或 LiDAR 模态解析 cache 行为
- **AND** 不包含 image 或 LiDAR 的组合 MUST 不需要相关 cache 参数才能启动

#### Scenario: profile 使用相同 policy
- **WHEN** 用户运行训练 I/O profile 入口
- **THEN** profile MUST 使用与训练入口一致的 cache policy 解析逻辑
- **AND** profile 输出 MUST 记录实际 cache policy 和 cache 目录

### Requirement: 场景化训练与评估输出
训练和默认评估流程 MUST 按 DeepSense6G 场景归类输出运行目录。默认输出根目录保持 `outputs`，DeepSenseG 运行目录 MUST 写入 `outputs/<scene_slug>/<run_name>/` 或等价的用户配置根目录下。

#### Scenario: 显式 Scenario 9 训练输出归档到 scene9
- **WHEN** 用户显式选择 Scenario 9 并运行训练且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scene9/<run_name>/`
- **AND** checkpoints、`final_config.yaml`、`train_log.json`、metrics、TensorBoard event 和训练曲线 MUST 都写入该运行目录

#### Scenario: 默认 Scenario 32 训练输出归档到 scene32
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 训练配置
- **THEN** 运行目录 MUST 创建在 `outputs/scene32/<run_name>/`
- **AND** 同名 Scenario 9 运行目录不得被覆盖

#### Scenario: resume 使用默认场景化运行目录
- **WHEN** 用户设置 `training.resume: true`、固定 `output.run_name` 且使用默认 Scenario 32
- **THEN** 系统 MUST 从 `outputs/scene32/<run_name>/checkpoints/last.pth` 恢复训练
- **AND** 系统不得回退到不同场景的同名运行目录

#### Scenario: 显式评估输出目录保持完整路径
- **WHEN** 用户通过评估入口显式传入 `--output-dir`
- **THEN** 系统 MUST 使用该目录作为完整输出目录
- **AND** 系统不得额外追加 `scene_slug`

### Requirement: 当前训练产物迁移到 Scenario 9
变更实施后，现有本地训练产物 MUST 被归类到 Scenario 9 输出目录。迁移 MUST 保留每个运行目录下的 checkpoint、日志、配置、metrics、TensorBoard 和 artifacts。

#### Scenario: 迁移现有运行目录
- **WHEN** 仓库中存在 `outputs/<run_name>/` 形式的历史训练目录
- **THEN** 迁移后该目录 MUST 位于 `outputs/scene9/<run_name>/`
- **AND** 原目录内容 MUST 保持完整

#### Scenario: 迁移现有最佳 checkpoint 目录
- **WHEN** 仓库中存在 `outputs/best_checkpoints/`
- **THEN** 迁移后历史 Scenario 9 registry MUST 位于 `outputs/scene9/best_checkpoints/`
- **AND** KD 配置默认解析 MUST 能找到迁移后的 teacher checkpoint

#### Scenario: 迁移避免覆盖
- **WHEN** `outputs/scene9/<run_name>/` 已经存在
- **THEN** 迁移 MUST 避免静默覆盖
- **AND** 系统 MUST 选择清晰的冲突处理方式或报告需要人工处理的冲突路径

### Requirement: 场景选择命令行覆盖
训练和评估入口 MUST 支持通过现有 dotted override 选择场景，不需要新增独立 CLI 参数。

#### Scenario: 命令行覆盖到 Scenario 9
- **WHEN** 用户运行 `python scripts/train.py --config <config> data.dataset.scene=9`
- **THEN** 系统 MUST 使用 Scenario 9 的数据默认值和输出目录分组
- **AND** 最终配置 MUST 记录覆盖后的场景
