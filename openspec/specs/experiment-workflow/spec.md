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
- **THEN** 系统 MUST 将 `ATop-3` 计算为所有 `J` 个未来目标时隙 Top-3 accuracy 的平均值
- **AND** 系统 MUST 将 `ATop-5` 计算为所有 `J` 个未来目标时隙 Top-5 accuracy 的平均值
- **AND** 系统 MUST 将 `ADBA` 计算为所有 `J` 个未来目标时隙 DBA 的平均值，且 DBA MUST 使用 Top-3 预测 beam 计算

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
训练、评估和 profile 入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的实际 cache 读写开关传递给 dataset。解析过程 MUST 使用配置中的启用模态，不得要求用户为每个单模态或 fusion 组合手动设置低层 cache 读写字段。系统 MUST 不再解析或传递 `image_motion_*` 低层开关。

#### Scenario: 单模态 image 训练不解析 image motion cache
- **WHEN** 用户运行 image-only 训练配置
- **THEN** 训练入口 MUST 使用 RGB/ImageNet image 输入构建 dataset
- **AND** 训练入口 MUST 不生成 `image_motion_use_cache` 或 `image_motion_write_cache`
- **AND** 用户 MUST 不能通过命令行恢复这些已删除低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 训练入口 MUST 只为该组合包含的受支持 cache 模态解析 cache 行为
- **AND** 不包含 LiDAR 的组合 MUST 不需要相关 cache 参数才能启动
- **AND** 包含 image 的组合 MUST 不需要且不得接受 image motion cache 参数

#### Scenario: profile 使用相同 policy
- **WHEN** 用户运行训练 I/O profile 入口
- **THEN** profile MUST 使用与训练入口一致的 cache policy 解析逻辑
- **AND** profile 输出 MUST 记录实际 cache policy 和受支持 cache 目录
- **AND** profile 输出 MUST 不记录 image motion cache 目录或读写开关

### Requirement: 场景化训练与评估输出
训练和默认评估流程 MUST 按 DeepSense6G 场景归类输出运行目录。默认输出根目录保持 `outputs`，DeepSense6G 运行目录 MUST 写入 `outputs/<scene_slug>/<run_name>/` 或等价的用户配置根目录下。

#### Scenario: 显式 Scenario 9 训练输出归档到 scene9
- **WHEN** 用户显式选择 Scenario 9 并运行训练且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scene9/<run_name>/`
- **AND** checkpoints、`final_config.yaml`、`train_log.json`、metrics、TensorBoard event 和训练曲线 MUST 都写入该运行目录

#### Scenario: 默认 Scenario 31 训练输出归档到 scene31
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 训练配置
- **THEN** 运行目录 MUST 创建在 `outputs/scene31/<run_name>/`
- **AND** 同名 Scenario 9 或 Scenario 32 运行目录不得被覆盖

#### Scenario: 显式 Scenario 32 训练输出归档到 scene32
- **WHEN** 用户显式选择 Scenario 32 并运行训练且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scene32/<run_name>/`
- **AND** 同名 Scenario 31 运行目录不得被覆盖

#### Scenario: resume 使用默认场景化运行目录
- **WHEN** 用户设置 `training.resume: true`、固定 `output.run_name` 且使用默认 Scenario 31
- **THEN** 系统 MUST 从 `outputs/scene31/<run_name>/checkpoints/last.pth` 恢复训练
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

### Requirement: 吞吐优化配置与日志
训练配置 MUST 暴露吞吐相关开关，包括 DataLoader worker/prefetch 参数、non-blocking transfer、AMP 和预处理 cache 读取/写入策略。训练日志、最终配置或 profile 输出 MUST 记录这些实际生效的吞吐参数，便于比较不同实验设置。

#### Scenario: 记录吞吐参数
- **WHEN** 一次训练或 profile 运行启动
- **THEN** 输出配置或日志 MUST 记录 `num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、non-blocking transfer、AMP enabled/dtype 和启用的 cache 目录
- **AND** 对启用 image 或 LiDAR 的配置 MUST 记录对应 cache 参数 hash 目录

#### Scenario: 并行实验默认不过度放大 worker
- **WHEN** 用户使用 canonical 单模态或 fusion YAML 运行实验
- **THEN** 配置 SHOULD 使用适合并行实验的保守 `num_workers` 和 `prefetch_factor`
- **AND** 用户 MUST 能通过命令行覆盖这些参数以寻找单实验最高吞吐

### Requirement: AMP 训练配置兼容
训练工作流 MUST 支持通过配置启用或关闭 AMP。AMP 配置 MUST 不影响 checkpoint 保存、早停、scheduler、TensorBoard、registry 和评估指标输出结构。

#### Scenario: 开启 AMP 完成短训练
- **WHEN** 用户在 CUDA device 上启用 AMP 并运行 1 epoch smoke test
- **THEN** 训练 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存
- **AND** 训练日志 MUST 记录 AMP 已启用和实际 dtype

#### Scenario: 关闭 AMP 保持旧行为
- **WHEN** 用户关闭 AMP 或在 CPU device 上运行训练
- **THEN** 训练 MUST 保持现有 FP32 行为
- **AND** 旧配置未声明 AMP 字段时 MUST 能继续运行

### Requirement: 虚拟 canonical 配置工作流
训练、评估和测试工作流 MUST 接受由配置加载器生成的虚拟 canonical fusion 配置。虚拟配置 MUST 在进入训练、评估、dry-run、override 合并、验证和 artifact 写出之前被解析为完整配置字典。

#### Scenario: 训练入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml`
- **THEN** 系统 MUST 解析该 canonical path 并启动 fusion logits KD 训练流程
- **AND** 训练流程 MUST 不要求 `configs/fusion/gps_mmwave_logits_kd.yaml` 在磁盘上存在

#### Scenario: 评估入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/evaluate.py --config configs/fusion/gps_mmwave_logits_kd.yaml --weights <path>`
- **THEN** 系统 MUST 解析该 canonical path 并构建对应 fusion student 模型
- **AND** 评估流程 MUST 只准备该配置启用的模态输入

#### Scenario: dry-run 使用虚拟 canonical 配置
- **WHEN** 用户运行 `python scripts/train.py --config configs/fusion/gps_mmwave_logits_kd.yaml --dry-run`
- **THEN** 系统 MUST 先生成 canonical 配置，再应用 dry-run 覆盖
- **AND** dry-run MUST 使用 synthetic dataset、单 epoch 和关闭 worker 的现有行为

#### Scenario: 保存完整 final config
- **WHEN** 使用虚拟 canonical 配置完成训练
- **THEN** 系统 MUST 在运行目录保存完整解析后的 `final_config.yaml`
- **AND** `final_config.yaml` MUST 包含训练复现所需的全部字段，而不是只保存虚拟路径或生成规则

#### Scenario: CLI override 覆盖虚拟配置
- **WHEN** 用户通过 `--override` 或点式未知参数覆盖虚拟 canonical 配置字段
- **THEN** 系统 MUST 在生成 canonical 配置之后应用这些覆盖
- **AND** 覆盖优先级 MUST 与实体 YAML 配置保持一致

### Requirement: 实验输出记录 split 协议
训练和评估流程 MUST 在运行产物中记录足够的 split 协议信息，用于判断不同实验是否使用同一数据协议并可横向比较。记录 MUST 包含实际 CSV 路径、样本数和 `balanced_seq` split metadata 路径或核心字段。

#### Scenario: 训练输出包含 split metadata 引用
- **WHEN** 训练入口构建 train/test dataset
- **THEN** `final_config.yaml`、`train_log.json` 或等价运行产物 MUST 记录 split metadata 路径或核心字段
- **AND** 记录 MUST 包含 split 策略、seed、train/test `seq_index` 数量和 train/test 样本数

#### Scenario: 评估输出包含 split 协议
- **WHEN** 评估入口构建 test dataset
- **THEN** 评估报告 MUST 记录实际使用的 test CSV 和可用的 split 协议信息
- **AND** 当当前 CSV 缺少 `balanced_seq` split metadata 时，系统 MUST 给出清晰错误或显式警告，避免把未知 split 协议误当成新协议结果

#### Scenario: 跨模态 split 可比较
- **WHEN** 用户使用同一组 train/test CSV 运行 image、radar、GPS、LiDAR、mmWave 或 fusion 实验
- **THEN** 各运行产物中的 split 协议信息 MUST 能显示它们使用相同 CSV 和相同 split metadata
- **AND** 如果 CSV 路径或 split metadata 不同，用户 MUST 能从运行产物中看出这些结果不应直接作为同一 split 协议比较

### Requirement: 未来标签时隙对齐
训练、验证、评估、诊断预测导出和 KD 相关 loss MUST 使用 `num_pred` 个未来标签时隙。`num_pred=3` 时，系统 MUST 将 label 和预测 slot 解释为 `[t+1, t+2, t+3]`，不得包含当前或历史最后一个 beam。

#### Scenario: 训练 loss 使用未来标签
- **WHEN** 训练流程准备 batch 且 `num_pred: 3`
- **THEN** loss 输入 logits MUST 与 `[t+1, t+2, t+3]` 三个标签时隙对齐
- **AND** flatten 后的 logits 数量 MUST 等于 flatten 后的 labels 数量

#### Scenario: 输出 slot 选择使用 future horizon
- **WHEN** 模型输出 logits 的时间维长度大于或等于 `num_pred`
- **THEN** 统一 slot 选择 helper MUST 返回最后 `num_pred` 个 slot
- **AND** 返回结果 MUST 与 `prepare_labels()` 输出的 future labels 同长
- **AND** 该 helper 的语义 MUST 表示长时序输出对齐，不得作为新 CRAF/MARF `num_pred + 1` fusion head 的兼容承诺

#### Scenario: 输出 slot 不足时报错
- **WHEN** 模型输出 logits 的时间维长度小于 `num_pred`
- **THEN** 训练、验证或评估流程 MUST 报出清晰错误
- **AND** 系统 MUST 不通过重复、padding 或拼接历史 beam 自动补齐 prediction slots

#### Scenario: 诊断预测导出保留 t+1
- **WHEN** viewer prediction export 写出 `confidence_curves` 或 `beam_distribution`
- **THEN** 导出的第一个 horizon MUST 表示 `t+1`
- **AND** 导出逻辑 MUST 不把第一个预测 slot 当作 current beam 丢弃

### Requirement: 训练流程支持 CRAF 输出适配
训练流程 MUST 能消费 CRAF/MARF dict 输出，同时保持现有三元组模型输出兼容。输出适配 MUST 提取 logits、训练 feature、蒸馏 feature 和可选 diagnostics；当 feature-based KD 或 diagnostics 需要真实 feature 时，系统 MUST 使用模型输出的真实 feature 字段，不得用 logits 伪装为 feature 静默继续。

#### Scenario: CRAF dict 输出训练
- **WHEN** 模型 forward 返回包含 `logits`、`input_features` 和 `output_features` 的 dict
- **THEN** 训练流程 MUST 从 dict 中提取 logits 计算任务 loss
- **AND** 训练流程 MUST 使用 dict 中的真实 feature 字段执行需要 feature 的 KD 或 diagnostics
- **AND** 训练流程 MUST 将非核心字段作为 diagnostics 传递给 CRAF/MARF 附加 loss 和日志路径

#### Scenario: dict 输出缺少 logits
- **WHEN** 模型 forward 返回 dict 但不包含受支持的 logits 字段
- **THEN** 输出适配 MUST 报错
- **AND** 训练、验证和诊断导出流程 MUST 不猜测其它 tensor 作为 logits

#### Scenario: 需要 feature 的路径缺少 feature
- **WHEN** 配置启用需要 `input_features` 或 `output_features` 的 KD、auxiliary diagnostics 或 feature 对齐路径
- **AND** 模型输出没有提供对应真实 feature
- **THEN** 训练流程 MUST 报错
- **AND** 系统 MUST 不使用 logits fallback 产生 feature-based loss

#### Scenario: 旧模型三元组输出训练
- **WHEN** 模型 forward 返回 `(pred, input_features, output_features)`
- **THEN** 训练流程 MUST 保持当前 loss、KD 和指标计算语义
- **AND** 三元组中的 feature MUST 被视为真实 feature 输入

#### Scenario: 输出 slot 截取精确对齐
- **WHEN** 模型输出 slot 数已经等于 `num_pred`
- **THEN** 训练流程 MUST 直接使用这些 slot 与未来标签对齐
- **AND** 不得因再次截取而改变语义

### Requirement: 训练流程支持 CRAF 附加 loss
训练流程 MUST 在 CRAF 显式配置时组合普通任务 loss、beam-aware soft label loss、单模态辅助 loss 和 counterfactual gate loss。未启用的 loss 权重 MUST 不影响总 loss。

#### Scenario: 只启用普通任务 loss
- **WHEN** CRAF 附加 loss 权重均为 0
- **THEN** 训练总 loss MUST 等于普通任务 loss 或现有 distiller 组合结果

#### Scenario: 启用 gate loss
- **WHEN** counterfactual gate supervision 产生 gate target
- **THEN** 训练流程 MUST 将 gate loss 按配置权重加入总 loss
- **AND** 日志 MUST 记录 gate loss 摘要

#### Scenario: ignore index 处理一致
- **WHEN** 标签中包含 `-100`
- **THEN** CRAF 附加 loss MUST 跳过这些位置
- **AND** 普通指标计算 MUST 保持现有 ignore index 语义

### Requirement: 评估流程支持 CRAF 输出
评估流程 MUST 能从 CRAF 输出中提取 beam logits 并计算现有 Top-K、DBA 和 loss 指标。评估流程 MUST 不执行训练专用 counterfactual forward。

#### Scenario: CRAF 模型评估
- **WHEN** 用户评估 CRAF checkpoint
- **THEN** 评估流程 MUST 提取 CRAF logits
- **AND** 评估流程 MUST 保存与现有模型一致的 metrics 文件

#### Scenario: 评估跳过 counterfactual
- **WHEN** 配置中 counterfactual training 曾启用
- **THEN** 评估流程 MUST 不执行 drop-forward gate supervision
- **AND** 评估结果 MUST 只反映正常 effective modality mask 下的预测表现

### Requirement: CRAF 日志与运行产物
训练输出 MUST 在 CRAF diagnostics 可用时保存 reliability、counterfactual 和 auxiliary loss 摘要，并 MUST 保持现有 `train_log.json`、`metrics.json` 和 TensorBoard 输出兼容。

#### Scenario: train_log 记录 CRAF 字段
- **WHEN** CRAF 训练完成至少一个 epoch
- **THEN** `train_log.json` MUST 包含 CRAF 附加 loss 和每模态 reliability 的 epoch 摘要

#### Scenario: final_config 保存 CRAF 配置
- **WHEN** CRAF 训练启动
- **THEN** `final_config.yaml` MUST 保存实际生效的 CRAF 模型、loss、counterfactual 和 modality dropout 配置

#### Scenario: 旧模型日志结构兼容
- **WHEN** 用户训练非 CRAF 模型
- **THEN** 输出日志 MUST 保持现有字段兼容
- **AND** CRAF 专属字段 MAY 缺省

### Requirement: CRAF smoke test 工作流
项目 MUST 提供可在 conda 环境中运行的 CRAF smoke test，覆盖模型构建、forward、loss、backward、验证和日志写入的核心路径。

#### Scenario: synthetic CRAF 短训练
- **WHEN** 开发者运行 CRAF synthetic 或小数据短训练测试
- **THEN** 训练流程 MUST 完成至少一个 optimizer step
- **AND** 验证流程 MUST 产出 metrics

#### Scenario: CRAF 配置加载测试
- **WHEN** 开发者运行配置加载测试
- **THEN** CRAF 示例配置和 baseline 示例配置 MUST 能通过 config loader 解析

### Requirement: CRAF 稳定化训练工作流
训练流程 MUST 支持 CRAF 稳定化训练配置，包括 warmup gate 固定、CE-only 反事实目标、ignore band、gate/loss schedule 和 softmax gate 诊断。

#### Scenario: warmup 阶段不扰动主任务训练
- **WHEN** CRAF 配置处于 warmup epoch
- **THEN** 训练流程 MUST 执行普通 forward、任务 loss、可配置的 warmup auxiliary loss 和优化步骤
- **AND** 训练流程 MUST 不执行会产生 gate target loss 的 counterfactual supervision

#### Scenario: 反事实启用后写入有效权重
- **WHEN** counterfactual supervision 已启用
- **THEN** `train_log.json` MUST 记录 gate loss 的目标权重和当前有效权重
- **AND** TensorBoard 启用时 MUST 写入等价标量

#### Scenario: 旧训练配置兼容
- **WHEN** CRAF 配置未提供新的稳定化字段
- **THEN** 训练流程 MUST 使用向后兼容默认值
- **AND** 非 CRAF 模型 MUST 不读取或依赖这些字段

### Requirement: CRAF 稳定化实验矩阵
项目 MUST 提供用于定位模态失衡问题的最小 CRAF 消融实验入口。

#### Scenario: token transformer 无 gate baseline
- **WHEN** 用户运行 token transformer 无 gate 配置
- **THEN** 模型 MUST 使用 CRAF tokenizer 与 Transformer backbone
- **AND** 训练流程 MUST 不启用 reliability gate 和 counterfactual gate loss

#### Scenario: CRAF 无反事实 baseline
- **WHEN** 用户运行 CRAF no-counterfactual 配置
- **THEN** 模型 MAY 构建 reliability estimator
- **AND** 训练流程 MUST 固定 gate 或跳过 counterfactual gate supervision

#### Scenario: 固定强模态 prior sanity check
- **WHEN** 用户运行固定 GPS/mmWave 高、image/LiDAR/radar 低的 prior 配置
- **THEN** 训练流程 MUST 使用该 prior 作为诊断 gate 或 dataset prior 输入
- **AND** 该配置 MUST 明确标记为 sanity check 而非默认算法

### Requirement: Teacher-prior CRAF stage workflow
训练流程 MUST 支持 teacher-prior CRAF 的 Stage 1、Stage 2 和 Stage 3 工作流，并 MUST 继续复用统一训练入口、输出目录、checkpoint、TensorBoard 和 `train_log.json` 语义。

#### Scenario: Stage 1 训练单模态 teacher
- **WHEN** 用户运行任一单模态 teacher-prior Stage 1 配置
- **THEN** 系统 MUST 使用对应单模态数据和 teacher 模型训练
- **AND** 输出目录 MUST 保存 best checkpoint、last checkpoint、最终配置和可供 teacher registry 读取的验证指标

#### Scenario: Stage 2 初始化发生在 optimizer 前
- **WHEN** 用户运行 Stage 2 teacher-init prior 配置
- **THEN** 系统 MUST 在构建 optimizer 前加载 teacher encoder 并应用冻结策略
- **AND** optimizer MUST 只包含 `requires_grad=True` 的参数

#### Scenario: Stage 3 checkpoint 加载后应用 finetune 策略
- **WHEN** 用户运行 Stage 3 selective fine-tuning 配置
- **THEN** 系统 MUST 先加载 Stage 2 checkpoint
- **AND** 系统 MUST 再应用选择性冻结/解冻策略并构建参数组 optimizer

### Requirement: Teacher registry build command
项目 MUST 提供可命令行运行的 teacher registry 构建流程。该流程 MUST 使用 conda 环境中的 Python 运行，并 MUST 能从配置或命令行参数指定 teacher root、输出路径、prior 模式和场景。

#### Scenario: 从 teacher 根目录生成 registry
- **WHEN** 用户运行 teacher registry 构建命令并指定 teacher root
- **THEN** 系统 MUST 扫描或读取五个单模态 teacher 输出目录
- **AND** 系统 MUST 写出 teacher registry JSON 到指定路径

#### Scenario: registry 写出路径父目录不存在
- **WHEN** 用户指定的 teacher registry 输出路径父目录不存在
- **THEN** 系统 MUST 创建父目录
- **AND** 系统 MUST 不覆盖 unrelated 输出文件

### Requirement: Teacher-prior CRAF optimizer 参数组
训练流程 MUST 支持 Stage 3 参数组 optimizer。参数组 MUST 按 fusion/head/gate/strong encoder/weak encoder 或等价角色划分，并 MUST 在训练日志中记录每组学习率和参数量。

#### Scenario: Stage 3 参数组非空
- **WHEN** Stage 3 配置解冻 GPS 和 mmWave encoder
- **THEN** strong encoder 参数组 MUST 包含 GPS 和 mmWave encoder 参数
- **AND** weak encoder 参数组 MUST 不包含 frozen image、radar 或 LiDAR 参数
- **AND** fusion、head 和 gate 参数组 MUST 非空

#### Scenario: 冻结参数不进入 optimizer
- **WHEN** 某个 encoder 参数 `requires_grad=False`
- **THEN** optimizer 参数组 MUST 不包含该参数
- **AND** 训练日志 MUST 记录该 encoder 为 frozen

### Requirement: Teacher-prior CRAF validation subsets
验证流程 MUST 支持对支持 force modality mask 的 fusion 模型运行显式模态组合评估。该能力 MUST 只在模型支持 force modality mask 且配置启用时运行，并 MUST 支持从 teacher prior 或等价配置中解析 top-prior、single-best-prior 和 low-prior 模态集合。既有 CRAF 配置 MUST 继续可用，MARF 配置 MUST 使用同一验证入口。

#### Scenario: 运行 prior-driven strong-only 和 weak-only 验证
- **WHEN** 配置启用 `evaluation.modality_subsets` 且 teacher prior 可用
- **THEN** 验证流程 MUST 使用 force modality mask 分别评估 strong-only 和 weak-only 组合
- **AND** strong-only MUST 对应当前 prior 最高的 top-k 可用模态
- **AND** weak-only MUST 对应当前 prior 最低的一组可用模态
- **AND** strong-only 和 weak-only 的实际模态列表 MUST 记录到验证输出或运行日志

#### Scenario: all subset 与官方验证一致
- **WHEN** 配置请求 `all` subset
- **THEN** `all` subset MUST 使用全部启用模态执行与官方 validation 等价的 forward
- **AND** `val/subset/all/top1` MUST 与官方 `accuracy/val` 在同一 checkpoint 和 dataloader 上一致

#### Scenario: 支持 MARF subset 名称
- **WHEN** 配置请求 `top_prior`、`single_best_prior`、`random_with_top_prior` 或 low-prior subset
- **THEN** 验证流程 MUST 按 prior 和配置参数解析对应 force mask
- **AND** 验证结果 MUST 包含每个成功评估 subset 的 Top-1、ATop-3、ATop-5、ADBA 和 loss

#### Scenario: 非 opt-in 模型跳过模态组合验证
- **WHEN** 模型不支持 `supports_force_modality_mask`
- **THEN** 验证流程 MUST 跳过模态组合评估
- **AND** 默认验证指标 MUST 仍正常产出

### Requirement: Teacher-prior CRAF smoke tests
项目 MUST 提供面向 teacher-prior CRAF 的短训练和定向测试路径。测试命令 MUST 使用 `conda run -n kd_mm_beam` 环境约束。

#### Scenario: PriorResidualGate 初始化测试
- **WHEN** 开发者运行 CRAF 定向测试
- **THEN** 测试 MUST 覆盖 prior residual gate 初始化后 gate 接近 prior
- **AND** 测试 MUST 覆盖 unavailable modality mask

#### Scenario: Stage 2/3 workflow smoke test
- **WHEN** 开发者运行 Stage 2 或 Stage 3 synthetic smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存
- **AND** 测试 MUST 验证冻结或选择性解冻策略生效

### Requirement: G2D training workflow
训练入口 MUST 支持通过配置启动 G2D 训练。G2D 训练 MUST 使用 fusion student 作为可训练主模型，MUST 使用多个 frozen 单模态 teacher，MUST 保存常规训练产物和 G2D diagnostics。

#### Scenario: 启动 G2D-lite 训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml`
- **THEN** 系统 MUST 构建 fusion student
- **AND** 系统 MUST 构建并冻结配置中的单模态 teacher ensemble
- **AND** 系统 MUST 使用 supervised CE、feature KD 和 logit KD 完成训练 step

#### Scenario: 启动 G2D-global 训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml`
- **THEN** 系统 MUST 执行 G2D 训练 step
- **AND** 系统 MUST 在 optimizer step 前应用 SMP 梯度屏蔽
- **AND** 训练日志或 diagnostics MUST 记录当前 active modalities

#### Scenario: 启动 G2D-horizon 诊断训练
- **WHEN** 用户运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_horizon.yaml`
- **THEN** 系统 MUST 运行 G2D 训练
- **AND** 每个 epoch diagnostics MUST 记录 `t+1`、`t+2` 和 `t+3` 的 modality ranking

### Requirement: Future horizon flat metrics
验证和评估输出 MUST 在现有 nested top-k 数组之外，增加 future horizon 扁平指标字段。字段 MUST 使用 `t1/t2/t3/avg` 命名，并 MUST 不输出历史 current beam 或 h0 指标。

#### Scenario: 保存三步 Top-K 扁平字段
- **WHEN** 验证阶段产出 logits `[B,3,64]` 和 labels `[B,3]`
- **THEN** `metrics.json` MUST 包含 `val_top1_t1`、`val_top1_t2`、`val_top1_t3` 和 `val_top1_avg`
- **AND** `metrics.json` MUST 包含 `val_top3_avg` 和 `val_top5_avg`
- **AND** 这些 avg 字段 MUST 对有效 future horizon 求平均

#### Scenario: 不输出旧 h0 指标
- **WHEN** G2D 或普通 future-only 评估写出 metrics
- **THEN** metrics MUST 不包含 `top1_h0`
- **AND** metrics MUST 不包含 `top1_future_avg`
- **AND** metrics MUST 不包含 `beam8_acc`

### Requirement: G2D validation commands
G2D 实现 MUST 提供定向测试和 smoke training 验证命令，并且所有 Python 命令 MUST 使用 `conda run -n kd_mm_beam`。

#### Scenario: 运行 G2D 定向测试
- **WHEN** 开发者验证 G2D 实现
- **THEN** 推荐测试命令 MUST 为 `conda run -n kd_mm_beam pytest -q tests/test_g2d_loss.py tests/test_g2d_distiller.py tests/test_g2d_smp.py tests/test_g2d_diagnostics.py`
- **AND** 测试 MUST 覆盖 loss shape、teacher confidence、SMP scheduler、gradient mask 和 diagnostics schema

#### Scenario: 运行 G2D smoke training
- **WHEN** 开发者完成 G2D 实现
- **THEN** 开发者 MUST 能使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_g2d_lite.yaml -o training.epochs=1`
- **AND** 该 smoke run MUST 完成 forward、loss、backward、optimizer step、validation 和 diagnostics 保存

### Requirement: Fusion 实验配置命名保持场景中立
推荐的 fusion 实验配置文件名、`experiment.name` 和 `output.run_name` MUST 不硬编码 `scene32_` 前缀。场景选择 MUST 通过 dataset 场景字段、命令行覆盖、输出根目录或 checkpoint metadata 表达，而不是混入方法 slug。

#### Scenario: MARF 主配置不包含 scene32 前缀
- **WHEN** 开发者加载推荐 MARF 主实验配置
- **THEN** 配置路径、`experiment.name` 和 `output.run_name` MUST 不包含 `scene32_`
- **AND** 配置 MAY 继续默认选择 Scene 32 数据集字段

#### Scenario: CRAF/MARF ablation 配置不包含 scene32 前缀
- **WHEN** 开发者加载推荐 CRAF 或 MARF ablation 配置
- **THEN** 配置文件名、`experiment.name` 和 `output.run_name` MUST 使用场景中立方法名
- **AND** 用户 MUST 能通过 dataset 场景覆盖在其它场景复用该方法配置

#### Scenario: 场景信息保留在数据和产物 metadata
- **WHEN** 训练或评估使用场景中立配置运行
- **THEN** dataset 配置和运行 metadata MUST 仍记录实际 scene / scene_id / scene_slug
- **AND** checkpoint registry MUST 能继续按场景目录或 metadata 区分产物

### Requirement: 默认 early stopping 指标使用 DBA
训练工作流 MUST 在 `experiment.objective: beam` 或未显式设置 objective 的历史 beam 训练中默认使用验证 DBA/ADBA 作为 early stopping 监控指标。objective-aware 非 beam 训练 MUST 使用对应预测目标的默认主指标：`occlusion` 使用 `val_occlusion_blocked_f1/max`，`position` 使用 `val_position_rmse/min`，`multitask` 使用 `val_multitask_loss/min` 或用户显式配置的可用 multitask 主指标。默认配置 MUST NOT 使用 `top1_val_acc`、`val_acc` 或其它 Top-1 验证准确率别名作为默认 early stopping 指标。

#### Scenario: 默认配置记录 DBA early stopping
- **WHEN** 用户使用未设置 `experiment.objective` 的默认 image、radar、GPS、LiDAR、mmWave 或 fusion 训练配置启动训练
- **THEN** 系统 MUST 将 objective 解析为 `beam`
- **AND** 系统 MUST 在解析后的最终配置中记录 early stopping 监控指标为 `val_adba` 或等价 DBA 别名
- **AND** 系统 MUST 将 early stopping 比较方向记录为越大越好
- **AND** 系统 MUST 不把 `top1_val_acc` 或等价 Top-1 验证准确率别名作为默认 early stopping 指标

#### Scenario: canonical beam 配置默认使用 DBA
- **WHEN** 开发者生成或读取 beam objective canonical 训练配置
- **THEN** canonical 配置 MUST 默认包含 DBA/ADBA early stopping 指标
- **AND** canonical 配置 MUST 不把 Top-1 验证准确率作为默认 early stopping 指标

#### Scenario: objective-aware occlusion 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: occlusion` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_occlusion_blocked_f1`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: max`

#### Scenario: objective-aware position 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: position` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_position_rmse`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`

#### Scenario: objective-aware multitask 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: multitask` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_multitask_loss`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`
- **AND** runtime metadata MUST 记录该 multitask loss 使用的分任务权重

#### Scenario: 显式覆盖 early stopping 指标
- **WHEN** 用户在训练配置或命令行覆盖中显式设置 early stopping 指标为 Top-1、loss 或其它受支持指标
- **THEN** 系统 MUST 使用用户显式指定的指标和比较方向
- **AND** 系统 MUST 校验该指标在当前 objective 的验证结果中真实可用
- **AND** 该覆盖 MUST 不改变项目默认配置继续使用 objective-specific 默认指标的要求

### Requirement: 训练循环按配置指标执行 early stopping
训练循环 MUST 从每个 epoch 的验证标量中解析配置的 early stopping 指标，并基于该指标更新最佳值、patience 计数和默认最佳 checkpoint。DBA/ADBA 和准确率类指标 MUST 按越大越好判断 improvement；loss 类指标 MUST 按越小越好判断 improvement。

#### Scenario: DBA improvement 重置 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 相比历史最佳值提升超过 `training.min_delta`
- **THEN** 系统 MUST 更新最佳 early stopping 值和最佳 epoch
- **AND** 系统 MUST 将 `epochs_without_improvement` 重置为 0
- **AND** 系统 MUST 保存默认最佳 checkpoint

#### Scenario: DBA 未提升累计 patience
- **WHEN** early stopping 指标为 `val_adba` 且当前 epoch 的 `val_adba` 未提升超过 `training.min_delta`
- **THEN** 系统 MUST 累加 `epochs_without_improvement`
- **AND** 当 `training.use_early_stopping` 启用且累计值达到 `training.patience` 时，系统 MUST 停止训练

#### Scenario: 缺失 DBA 指标时报错
- **WHEN** 默认 early stopping 指标为 DBA/ADBA 但验证结果没有产出可解析的 DBA/ADBA 标量
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 early stopping 指标，并提示用户补齐 DBA 指标或显式配置其它受支持指标

### Requirement: early stopping metadata 可复现
训练产物 MUST 记录实际使用的 early stopping 指标、比较方向、最佳值、最佳 epoch 和未提升 epoch 计数。恢复训练 MUST 优先使用这些通用 metadata 继续 early stopping 状态；历史 checkpoint 缺少通用 metadata 时，系统 MUST 使用兼容路径恢复已有 loss 或 Top-1 相关状态。

#### Scenario: checkpoint 记录 early stopping 状态
- **WHEN** 训练完成至少一个 epoch 并保存 `last.pth`
- **THEN** checkpoint metadata MUST 包含实际 early stopping 指标、比较方向、最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 运行日志或最终配置 MUST 能追溯本次训练使用的 early stopping 指标

#### Scenario: 恢复 DBA early stopping 状态
- **WHEN** 用户从包含通用 early stopping metadata 的 checkpoint 恢复训练
- **THEN** 系统 MUST 恢复 DBA/ADBA 的最佳值、最佳 epoch 和 `epochs_without_improvement`
- **AND** 后续 early stopping 判断 MUST 延续恢复前的指标和比较方向

#### Scenario: 兼容历史 checkpoint
- **WHEN** 用户从缺少通用 early stopping metadata 的历史 checkpoint 恢复训练
- **THEN** 系统 MUST 尽可能从历史 `best_val_loss`、`best_val_top1` 或等价字段恢复状态
- **AND** 系统 MUST 不因缺少新 metadata 而拒绝恢复历史 checkpoint

### Requirement: 默认实验记录 encoder 和 preprocessing profile
训练、验证和评估流程 MUST 在运行产物中记录 camera encoder 与 LiDAR preprocessing profile，使不同单模态 baseline 的结果可以横向比较。

#### Scenario: 记录 image encoder profile
- **WHEN** 一次 image-only 或包含 image 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 image profile、image encoder 类型、是否使用预训练权重、权重名称、freeze 策略和实际可训练 stage

#### Scenario: 记录 LiDAR preprocessing profile
- **WHEN** 一次 LiDAR-only 或包含 LiDAR 的 fusion 训练启动
- **THEN** final_config 或运行 metadata MUST 记录 LiDAR normalization、cache、ROI、FoV、ground/background filter 和安全增强配置

### Requirement: 单模态 baseline 回归检查
项目 MUST 提供面向 image 和 LiDAR 默认 baseline 的回归检查，防止默认配置重新退回到从头训练 camera encoder 或 LiDAR 多数类退化路径。

#### Scenario: image 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 image teacher/no-KD 配置使用 `resnet18_imagenet_rgb`
- **AND** 测试 MUST 验证该 encoder 配置启用 ImageNet 预训练权重

#### Scenario: LiDAR 默认配置回归检查
- **WHEN** 开发者运行配置测试
- **THEN** 测试 MUST 验证默认 LiDAR teacher/no-KD 配置显式启用 LiDAR streaming stats normalization
- **AND** 测试 MUST 验证该配置记录可追踪的 BEV ROI/cache 参数

#### Scenario: LiDAR 退化报告回归检查
- **WHEN** 开发者运行 LiDAR 评估或诊断测试
- **THEN** 输出报告 MUST 包含 majority-class baseline
- **AND** 输出报告 MUST 包含 LiDAR input quality summary
- **AND** 报告 MUST 能标记模型未超过 majority-class baseline 的退化风险

### Requirement: 训练流程支持多任务辅助 loss
训练流程 MUST 在保持现有 beam/KD 基础 loss 的前提下支持可选多任务辅助 loss。辅助 loss MUST 只在配置启用且 batch/model 均提供对应字段时计算；否则训练流程 MUST 保持现有 beam-only 行为。

#### Scenario: no-KD 多任务训练
- **WHEN** 用户运行启用遮挡和位置辅助任务的 no-KD fusion 训练
- **THEN** 训练流程 MUST 计算 beam CE、遮挡 BCE 和位置 MSE
- **AND** optimizer step MUST 使用三者加权后的总 loss
- **AND** train log MUST 记录每个 loss 分量

#### Scenario: KD 多任务训练
- **WHEN** 用户运行启用辅助任务的 logits KD 或 RKD fusion 训练
- **THEN** 训练流程 MUST 保留既有 KD 基础 loss 计算
- **AND** 训练流程 MUST 将辅助 loss 加到 student 总 loss
- **AND** teacher 模型 MUST 不被要求输出辅助头，除非配置显式启用 teacher auxiliary supervision

#### Scenario: 辅助字段缺失
- **WHEN** 配置启用辅助 loss 但 batch 或模型输出缺少必要字段
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的是 dataset target 还是 model auxiliary output

### Requirement: 验证和评估输出辅助指标
验证和评估流程 MUST 在启用多任务辅助监督时输出遮挡和位置指标，同时保留现有 Top-K、DBA、loss、degradation baseline 和 modality subset 评估语义。

#### Scenario: 验证输出遮挡指标
- **WHEN** 验证流程收到 `occlusion_logits` 和 `occlusion_label`
- **THEN** validation metrics MUST 包含遮挡 accuracy 和 blocked-class F1
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: 验证输出位置指标
- **WHEN** 验证流程收到 `position` 和 `position_target`
- **THEN** validation metrics MUST 包含 position RMSE
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: beam 指标保留
- **WHEN** 多任务辅助监督启用
- **THEN** 验证和评估流程 MUST 继续输出 beam Top-K、DBA、ATop-3、ATop-5 和 ADBA
- **AND** early stopping 默认 MUST 继续支持现有 `val_adba` 配置

### Requirement: 多任务运行产物可复现
训练和评估流程 MUST 在运行产物中记录多任务配置、遮挡阈值、辅助目标统计、loss 权重和辅助指标，确保后续评估和复现实验能加载相同的标签生成状态。

#### Scenario: final config 记录多任务状态
- **WHEN** 训练启用多任务辅助监督
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录遮挡阈值、阈值分位数、位置目标来源和 loss 权重
- **AND** checkpoint 或 normalization artifacts MUST 记录独立评估所需的辅助目标统计

#### Scenario: train log 记录辅助指标历史
- **WHEN** 训练至少完成一个 epoch 且启用多任务辅助监督
- **THEN** `train_log.json` MUST 包含遮挡和位置指标历史
- **AND** `training_outputs.npz` MUST 保存可画曲线的辅助 loss 或指标数组

### Requirement: Objective-aware 训练流程
训练流程 MUST 根据 `experiment.objective` 选择主 target、主模型输出、主 loss 和训练日志字段。`experiment.task` MUST 继续决定输入路由和模型 forward 路径。

#### Scenario: fusion occlusion 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: occlusion` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用遮挡 logits 和遮挡标签计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion position 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: position` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用位置输出和位置目标计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion multitask 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: multitask` 的训练配置
- **THEN** trainer MUST 计算 beam、occlusion 和 position 三个 loss 分量
- **AND** trainer MUST 按配置权重合成总 loss

### Requirement: Objective-aware 验证和评估
验证和评估流程 MUST 根据 `experiment.objective` 输出当前目标的主 metrics，并保留可计算的诊断 metrics。主 metrics MUST 支持 checkpoint 选择和 standalone evaluate。

#### Scenario: occlusion 验证指标
- **WHEN** 验证 `experiment.objective: occlusion` 的模型
- **THEN** validator MUST 输出遮挡 loss、accuracy 和 blocked-class F1
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_occlusion_blocked_f1`

#### Scenario: position 验证指标
- **WHEN** 验证 `experiment.objective: position` 的模型
- **THEN** validator MUST 输出位置 loss、RMSE 和 MAE
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_position_rmse`

#### Scenario: multitask 验证指标
- **WHEN** 验证 `experiment.objective: multitask` 的模型
- **THEN** validator MUST 输出 beam、occlusion 和 position 的分任务 metrics
- **AND** validator MUST 输出 multitask 总 loss 或配置指定的主指标

### Requirement: Objective-aware checkpoint registry
checkpoint registry 和 final config MUST 记录 objective-aware 指标，确保后续 evaluation 能按训练目标解释 checkpoint。

#### Scenario: 归档 occlusion checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: occlusion` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和遮挡指标
- **AND** evaluate MUST 能读取 registry artifact 并复用遮挡阈值

#### Scenario: 归档 position checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: position` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和位置指标
- **AND** evaluate MUST 能读取 registry artifact 并复用位置 target scaler

### Requirement: Objective metrics 可用性语义
训练、验证和评估流程 MUST 区分 active objective metrics 与 inactive metrics。未启用、缺少 head、缺少 target 或未实际计算的任务指标 MUST 不被写成 `0.0` 真实性能；系统 MUST 用缺失、`null`、`NaN` 或显式 availability metadata 表示不可用状态。

#### Scenario: beam-only 训练不写 position 零曲线
- **WHEN** 用户运行 `experiment.objective: beam` 且未启用 position target/head 的训练
- **THEN** TensorBoard MUST 不写入 `position/rmse` 或 `position/mae` 标量曲线
- **AND** epoch log MUST 不把 `val_position_rmse` 或 `val_position_mae` 记录为真实 `0.0`

#### Scenario: occlusion-only 训练不写 position 零曲线
- **WHEN** 用户运行 `experiment.objective: occlusion` 且未启用 position target/head 的训练
- **THEN** TensorBoard MUST 不写入 `position/rmse` 或 `position/mae` 标量曲线
- **AND** `training_outputs.npz` 若保留 position metric 数组 key，inactive slot MUST 使用 `NaN` 或等价不可用表示

#### Scenario: position-only 训练不写 occlusion 零曲线
- **WHEN** 用户运行 `experiment.objective: position` 且未启用 occlusion target/head 的训练
- **THEN** TensorBoard MUST 不写入 `occlusion/accuracy` 或 `occlusion/blocked_f1` 标量曲线
- **AND** epoch log MUST 不把 `val_occlusion_accuracy` 或 `val_occlusion_blocked_f1` 记录为真实 `0.0`

#### Scenario: multitask 训练写入全部 active metrics
- **WHEN** 用户运行 `experiment.objective: multitask` 且 beam、occlusion 和 position metrics 均可计算
- **THEN** TensorBoard MUST 写入 beam、occlusion 和 position 对应的 active scalar 曲线
- **AND** `train_log.json` MUST 记录三个任务的验证指标和 multitask 加权总 loss

#### Scenario: early stopping 不接受 inactive metric
- **WHEN** 用户配置的 early stopping metric 对当前 objective 不可用
- **THEN** 训练流程 MUST 在保存 misleading checkpoint 前抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 metric，并提示用户选择当前 objective 可用的 metric

### Requirement: Objective-aware validation 输出
验证和评估输出 MUST 只把真实计算的 auxiliary metrics 提升为 top-level metric，并 MUST 提供足够 metadata 说明哪些 objective targets 和 heads 已启用。inactive metric 不得通过默认零值绕过下游 early stopping 和图表解释。

#### Scenario: metrics JSON 省略 inactive auxiliary metric
- **WHEN** 验证 `experiment.objective: beam` 且未计算 position metric
- **THEN** `metrics.json` MUST 不把 top-level `val_position_rmse` 写成 `0.0`
- **AND** 输出 MUST 能通过 objective metadata 表明 position 不是本次 enabled head/target

#### Scenario: metrics JSON 包含 active position metric
- **WHEN** 验证 `experiment.objective: position` 且 position output、target 和 valid mask 均可用
- **THEN** `metrics.json` MUST 包含真实计算的 `val_position_rmse`
- **AND** 该值 MUST 用 position target scaler 反归一化后的尺度计算

#### Scenario: metrics JSON 包含 active occlusion metric
- **WHEN** 验证 `experiment.objective: occlusion` 且 occlusion logits、label 和 valid mask 均可用
- **THEN** `metrics.json` MUST 包含真实计算的 `val_occlusion_blocked_f1`
- **AND** 该值 MUST 可作为 `val_occlusion_blocked_f1/max` early stopping 来源

### Requirement: Beam TensorBoard 指标命名空间
训练流程 MUST 为 beam 预测写入 objective-specific TensorBoard 标量命名空间。`beam/*` 标量 MUST 只表示 active beam objective 或 multitask 中的 active beam 分任务，不得包含 occlusion-only 或 position-only 训练中的诊断性 beam accuracy。默认 TensorBoard 输出 MUST 不再依赖通用 `accuracy/*` 分组作为 beam 指标入口；历史通用 tag 只能作为显式兼容路径写入。

#### Scenario: beam objective 写入 beam 指标
- **WHEN** 用户运行 `experiment.objective: beam` 或未显式设置 objective 的历史 beam 训练，并启用 TensorBoard
- **THEN** 训练流程 MUST 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`
- **AND** 这些 tag MUST 分别对应当前 epoch 的 `train_acc`、`val_acc`、`val_atop3`、`val_atop5` 和 `val_adba`
- **AND** 写入前 MUST 跳过缺失、`null`、`NaN` 或非 finite 的值

#### Scenario: occlusion 单任务不污染 beam 指标
- **WHEN** 用户运行 `experiment.objective: occlusion` 的单任务训练，并启用 TensorBoard
- **THEN** 训练流程 MUST NOT 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 或 `beam/val_adba`
- **AND** 即使 validator 能计算诊断性 beam `val_acc`，该值也 MUST NOT 出现在 `beam/*` TensorBoard 命名空间中

#### Scenario: position 单任务不污染 beam 指标
- **WHEN** 用户运行 `experiment.objective: position` 的单任务训练，并启用 TensorBoard
- **THEN** 训练流程 MUST NOT 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 或 `beam/val_adba`
- **AND** position TensorBoard 指标 MUST 继续通过 `position/rmse` 和 `position/mae` 表示

#### Scenario: multitask 写入 active beam 分任务指标
- **WHEN** 用户运行 `experiment.objective: multitask` 且 beam 分任务参与 loss 或主验证指标计算，并启用 TensorBoard
- **THEN** 训练流程 MUST 写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`
- **AND** 训练流程 MUST 继续写入 active 的 `occlusion/*` 和 `position/*` 指标

#### Scenario: 默认不写历史通用 accuracy tag
- **WHEN** 用户启用 TensorBoard 且未显式设置 `output.tensorboard.legacy_accuracy_tags: true`
- **THEN** 训练流程 MUST NOT 写入 `accuracy/train`、`accuracy/val`、`accuracy/val_atop3`、`accuracy/val_atop5` 或 `dba/val_adba` 作为默认 beam 指标
- **AND** `train_log.json`、`training_outputs.npz` 和 checkpoint metadata MUST 继续保留既有内部 metric key，便于旧分析脚本读取

#### Scenario: 显式启用历史通用 tag
- **WHEN** 用户设置 `output.tensorboard.legacy_accuracy_tags: true` 并启用 TensorBoard
- **THEN** 训练流程 MAY 额外写入历史 `accuracy/*` 和 `dba/val_adba` tag
- **AND** 这些 legacy tag MUST 被文档标记为兼容入口，不得作为 objective-aware 实验比较的推荐入口

### Requirement: Beam metric alias 兼容
训练流程 MUST 支持 objective-specific beam metric 名称作为 early stopping 和用户配置别名。新增 `beam/*` 别名 MUST 解析到既有内部 metric key，同时历史 `accuracy/*` 和 `dba/*` 别名 MUST 保持可用。

#### Scenario: 使用 beam ADBA tag 配置 early stopping
- **WHEN** 用户将 early stopping metric 配置为 `beam/val_adba`
- **THEN** 系统 MUST 将该配置解析为内部 `val_adba`
- **AND** 比较方向 MUST 支持按 DBA/ADBA 语义使用越大越好

#### Scenario: 使用 beam Top-1 tag 配置 early stopping
- **WHEN** 用户将 early stopping metric 配置为 `beam/accuracy_val` 或 `beam/val_top1`
- **THEN** 系统 MUST 将该配置解析为内部 `val_acc`
- **AND** 比较方向 MUST 支持按 accuracy 语义使用越大越好

#### Scenario: 历史 early stopping 别名继续可用
- **WHEN** 用户将 early stopping metric 配置为 `accuracy/val`、`accuracy/val_top1` 或 `dba/val_adba`
- **THEN** 系统 MUST 继续解析到对应内部 beam metric
- **AND** 解析行为 MUST 不要求 TensorBoard 继续写入同名 legacy tag

