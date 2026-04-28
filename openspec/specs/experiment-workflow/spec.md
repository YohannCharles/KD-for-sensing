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
训练和评估流程 MUST 将运行产物写入统一输出目录。输出目录 MUST 至少包含本次运行的有效配置、checkpoint 或权重引用、metrics、训练曲线或日志，以及测试报告。训练流程 MUST 在启用 TensorBoard 时写入可由 TensorBoard 读取的标量 event 日志，并且 MUST 支持通过配置关闭该日志写入。训练流程 MUST 在启用进度显示时提供 `tqdm` 训练进度条，并且 MUST 将每个 epoch 的进度摘要保存到运行日志。

#### Scenario: 训练完成后保存运行配置
- **WHEN** 一次训练任务启动并创建输出目录
- **THEN** 系统 MUST 保存解析和覆盖后的最终配置，便于后续复现实验

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

### Requirement: 训练与评估行为等价
结构重构后，默认 image-only 和 image+radar 工作流 MUST 通过新脚本保持当前算法的核心行为语义，包括默认序列长度、预测步数、类别数、KD 模式、teacher 权重选择、student 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认超参数语义，并保持相同的任务类型

#### Scenario: 默认 student 架构
- **WHEN** 用户使用默认 image-only 或 image+radar student 实验配置构建模型
- **THEN** 系统 MUST 为 image-only 工作流构建轻量 `image_student`
- **AND** 系统 MUST 为 image+radar 工作流构建轻量 `fusion_student`
- **AND** 默认 student 配置 MUST 与仓库提供的对应 `All_models/*Std*.pth` 权重结构兼容

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径

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

