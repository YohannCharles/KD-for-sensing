# experiment-workflow Specification

## Purpose
定义配置驱动训练、评估、预处理、诊断、运行产物保存、README 入口边界以及 virtual/overlay 配置复现实验的工作流要求。
## Requirements
### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、`model.primary` 主模型、supervised/adaptation/JEPA/CSI 或诊断目标、训练超参数、优化器、调度器、输出目录、随机种子、GPS 特征模式和 fusion 模态选择。当前支持的训练配置 MUST 不覆盖 KD 模式或 teacher checkpoint；旧 KD、teacher/student no-KD、Hist、Top8 standalone、residual、camera residual、BGAM 和 viewer manifest 路径 MUST 在配置解析或 registry 层被拒绝。

#### Scenario: 使用配置启动 image-only 训练
- **WHEN** 用户通过当前 CLI 传入 image-only 训练配置
- **THEN** 系统 MUST 构建 image-only dataset、`model.primary`、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 使用配置启动 fusion 训练
- **WHEN** 用户通过当前 CLI 传入 fusion 训练配置
- **THEN** 系统 MUST 构建同时包含启用模态输入的 dataset、fusion `model.primary`、loss、optimizer 和 scheduler
- **AND** 系统 MUST 不要求 teacher checkpoint

#### Scenario: 使用配置启动 radar-only 训练
- **WHEN** 用户通过当前 CLI 传入 radar-only 训练配置
- **THEN** 系统 MUST 构建包含 radar 输入的 dataset、配置指定的 radar primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求 image 输入、teacher checkpoint 或 distiller

#### Scenario: 使用配置启动 GPS-only 训练
- **WHEN** 用户通过当前 CLI 传入 GPS-only 训练配置
- **THEN** 系统 MUST 构建包含 GPS 输入的 dataset、配置指定的 GPS primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** GPS 输入 MUST 使用配置声明的当前 GPS feature mode 和 train-split normalization

#### Scenario: 使用配置启动单模态评估
- **WHEN** 用户通过当前 CLI 传入 image、radar、GPS、LiDAR、mmWave 或 CSI 单模态评估配置和模型权重
- **THEN** 系统 MUST 构建配置指定的 primary model 并只使用启用模态完成评估
- **AND** 系统 MUST 保存当前 objective 支持的 Top-K、DBA、loss 或诊断指标

#### Scenario: 使用配置启动可选模态 fusion 训练
- **WHEN** 用户通过当前 CLI 传入带 `modalities` 的 fusion 配置
- **THEN** 系统 MUST 只准备并融合 `modalities` 中列出的当前支持模态
- **AND** 未启用模态的文件缺失 MUST 不阻止当前任务启动

#### Scenario: 使用当前 JEPA、GPS、CSI 和诊断 workflow
- **WHEN** 用户运行当前 JEPA pretraining/downstream、GPS-query pooling、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark 或其它 current benchmark 配置
- **THEN** 系统 MUST 使用对应 current workflow 的 `model.primary`、runner manifest 或诊断 schema
- **AND** 系统 MUST 不恢复 legacy KD、Hist、standalone Top8 selector、GPS residual、camera residual、BGAM 或 viewer manifest runtime

### Requirement: 命令行覆盖配置
实验入口 MUST 支持在命令行覆盖配置值。当前 CLI MUST 支持显式传入配置文件和关键参数覆盖；旧脚本 argparse 参数不得作为兼容入口保留，只能作为迁移默认值参考。命令行覆盖 MUST 不能绕过当前配置解析 guard 来重新启用 KD、teacher checkpoint、retired config alias 或旧研究路线。

#### Scenario: 覆盖训练轮数
- **WHEN** 用户通过命令行将训练轮数覆盖为 `1`
- **THEN** 系统 MUST 使用覆盖后的训练轮数，而不是配置文件中的默认训练轮数

#### Scenario: 覆盖当前实验参数
- **WHEN** 用户通过命令行覆盖 batch size、learning rate、scene、output run name、GPS feature mode、difficulty profile 或 manifest path
- **THEN** 系统 MUST 在最终配置、运行 metadata 或输出 manifest 中记录覆盖后的值
- **AND** 相对路径 MUST 继续按项目根目录解析

#### Scenario: 拒绝 KD 模式覆盖
- **WHEN** 用户通过命令行覆盖 `kd_mode`、`distillation.*`、`teacher_model_name`、`logits_kd`、`rkd`、`teacher_no_kd` 或 `student_no_kd`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向当前 `model.primary`、supervised/adaptation、JEPA、CSI、诊断或保留 baseline 入口

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
- **AND** 进度条 MUST 展示当前 epoch、batch 进度、训练损失、主任务损失、训练准确率和学习率中的关键状态

#### Scenario: 训练完成后保存进度日志
- **WHEN** 一次训练任务完成至少一个 epoch
- **THEN** 系统 MUST 在当前运行目录的训练日志中保存 epoch 级进度摘要
- **AND** 进度摘要 MUST 包含 epoch 编号、训练损失、训练主任务损失、训练准确率、验证损失、验证准确率和学习率
- **AND** 进度摘要 MUST 不包含新的训练蒸馏损失字段
- **AND** 日志保存 MUST 保持既有历史指标数组兼容

#### Scenario: 通过配置关闭 tqdm 进度显示
- **WHEN** 用户在训练配置中关闭进度显示
- **THEN** 系统 MUST 不创建可视化 `tqdm` 进度条
- **AND** 系统 MUST 继续保存训练日志和 epoch 级进度摘要

#### Scenario: 训练过程中写入 TensorBoard 标量日志
- **WHEN** 一次训练任务完成至少一个 epoch 且 TensorBoard 日志启用
- **THEN** 系统 MUST 在当前运行目录下写入 TensorBoard event 文件
- **AND** event 文件 MUST 记录训练总损失、训练主任务损失、训练准确率、验证损失、验证准确率、学习率、验证 `ATop-3`、验证 `ATop-5` 和验证 `ADBA` 标量
- **AND** event 文件 MUST 不新增 `loss/distillation` 或 KD 标量

#### Scenario: TensorBoard 记录跨时隙平均验证指标
- **WHEN** 一次训练 epoch 的验证阶段产出 per-slot Top-K accuracy 和 DBA 结果
- **THEN** 系统 MUST 将 `ATop-3` 计算为所有 `J` 个未来目标时隙 Top-3 accuracy 的平均值
- **AND** 系统 MUST 将 `ATop-5` 计算为所有 `J` 个未来目标时隙 Top-5 accuracy 的平均值
- **AND** 系统 MUST 将 `ADBA` 计算为所有 `J` 个未来目标时隙 DBA 的平均值，且 DBA MUST 使用 `K=3`、默认 `Δ=5`，按 `Y_1` 到 `Y_3` 的平均计算

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
结构重构后，默认 image-only、radar-only、GPS-only、LiDAR-only、mmWave 和 fusion 工作流 MUST 通过当前 CLI 保持核心训练、验证和评估语义，包括默认序列长度、预测步数、类别数、`model.primary` 架构选择、early stopping、gradient clipping、checkpoint 恢复和指标计算。历史 teacher/student 参数只可作为 checkpoint 读取、迁移 guard 或已明确标记的兼容背景出现，不得要求当前训练流程构建成对 KD runtime。

#### Scenario: 新配置默认参数
- **WHEN** 用户使用新脚本和默认配置启动训练或评估
- **THEN** 系统 MUST 使用从旧实现迁移而来的默认任务语义，并保持相同的任务类型
- **AND** `configs/<modality>/{strong,lightweight,supervised}.yaml` MUST 使用 `model.primary`
- **AND** current lightweight/supervised 配置 MUST 不依赖 `model.teacher`、`model.student`、`distillation.*` 或 teacher checkpoint
- **AND** 历史 teacher/student GRU 参数 MAY 在原代码兼容说明或墓碑 spec 中保留，但 MUST 标记为历史兼容而不是当前训练入口

#### Scenario: 默认 primary 架构与 GRU 层数
- **WHEN** 用户使用默认 image-only、radar-only、GPS-only、LiDAR-only、mmWave 或 fusion 当前实验配置构建模型
- **THEN** 系统 MUST 按 `model.primary.type` 构建对应 strong、lightweight、supervised、JEPA、CSI 或 baseline/control 模型
- **AND** current lightweight primary 的 GRU/temporal 参数 MUST 由其配置显式声明
- **AND** 文档 MUST 说明二层 GRU teacher/student 是历史 canonical 配置或特定兼容背景，不是当前默认结构

#### Scenario: checkpoint 恢复语义
- **WHEN** 用户在训练配置中启用 `training.resume`
- **THEN** 训练流程 MUST 尝试恢复 checkpoint
- **AND** 恢复 MUST 包含模型权重、optimizer、scheduler、已完成 epoch 和 best validation loss
- **AND** `training.start_epoch` MUST 不再是唯一影响恢复 epoch 的字段

#### Scenario: dry-run 训练
- **WHEN** 开发者使用 synthetic 或小比例数据运行一次短训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存的核心路径
- **AND** 使用目标兼容配置时，smoke test MUST 使用与该配置匹配的 GRU 层数构建模型

### Requirement: Radar-only KD 实验配置已移除
项目 MUST 不再提供 radar-only KD 配置。旧 `logits_kd` 和 `rkd` 配置 MUST 在配置解析阶段失败，并引导用户使用 `configs/radar/strong.yaml`、`configs/radar/lightweight.yaml` 或 `configs/radar/supervised.yaml`。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/logits_kd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 使用 RKD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/rkd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

#### Scenario: 旧 RadarTeacher checkpoint 自动解析被移除
- **WHEN** 用户运行当前 radar 训练配置
- **THEN** 系统 MUST 不解析 teacher checkpoint
- **AND** 训练流程 MUST 只更新 `model.primary`

#### Scenario: 旧 RadarTeacher checkpoint override 被拒绝
- **WHEN** 用户通过命令行覆盖 `distillation.teacher_model_name`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向当前 supervised/adaptation 入口

### Requirement: RadarStudent legacy no-KD 请求迁移
项目 MUST 拒绝旧 `configs/radar/student_no_kd.yaml` 入口，并将其解释为历史 no-KD/student 路径的 migration guard。当前 radar 轻量或 supervised 实验 MUST 使用 `configs/radar/lightweight.yaml`、`configs/radar/supervised.yaml` 或等价 `model.primary` 配置。

#### Scenario: 旧 RadarStudent no-KD 请求迁移
- **WHEN** 用户通过训练入口传入已退役的 `configs/radar/student_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该旧入口
- **AND** 错误信息 MUST 指向当前 radar lightweight 或 supervised 配置

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
项目 MUST 提供可选模态 fusion 配置，使用户能通过 `modalities` 手动选择当前支持模态的任意合法非空组合。Fusion 配置 MUST 构建单个 `model.primary`，不得要求 teacher/student 成对模型或 Fusion KD 配置。

#### Scenario: 运行 image+gps fusion
- **WHEN** 用户运行 `modalities: ["image", "gps"]` 的 fusion 配置
- **THEN** 系统 MUST 构建只包含 image 和 gps 分支的 fusion primary model
- **AND** 系统 MUST 不要求 radar 输入、teacher checkpoint 或 distiller

#### Scenario: 运行 radar+gps fusion
- **WHEN** 用户运行 `modalities: ["radar", "gps"]` 的 fusion 配置
- **THEN** 系统 MUST 构建只包含 radar 和 gps 分支的 fusion primary model
- **AND** 系统 MUST 不要求 image 输入

#### Scenario: 运行多模态 fusion
- **WHEN** 用户运行包含 image、radar、gps、lidar、mmwave 或 csi 的合法 fusion 配置
- **THEN** 系统 MUST 构建配置声明的启用模态输入和 fusion primary model
- **AND** 系统 MUST 使用统一训练、验证和评估流程输出指标

### Requirement: LiDAR 配置驱动实验
项目 MUST 支持通过配置文件启动 LiDAR-only 训练和评估。LiDAR-only 配置 MUST 使用当前 LiDAR dataset、preprocessing/cache contract、`model.primary`、统一训练/验证/评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 LiDAR-only 训练
- **WHEN** 用户通过当前 CLI 传入 LiDAR-only 训练配置
- **THEN** 系统 MUST 构建包含 LiDAR 输入的 dataset、配置指定的 LiDAR primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求 image、radar、GPS、teacher checkpoint 或 distiller
- **AND** LiDAR 输入 MUST 使用当前配置声明的 BEV、streaming stats 或 raw point cloud profile

#### Scenario: 使用配置启动 LiDAR-only 评估
- **WHEN** 用户通过当前 CLI 传入 LiDAR-only 评估配置和 LiDAR 模型权重
- **THEN** 系统 MUST 构建配置指定的 LiDAR primary model 并只使用 LiDAR 输入完成评估
- **AND** 系统 MUST 保存当前 metric profile 声明的 Top-K、DBA、loss 或诊断指标

### Requirement: LiDAR fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 LiDAR。包含 LiDAR 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程，并 MUST 构建单个 fusion primary model。

#### Scenario: 使用配置启动 image+radar+gps+lidar fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar"]` 的 fusion 配置
- **THEN** 系统 MUST 构建四个模态输入所需的 dataset 字段和 fusion primary model
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps 和 lidar 输入

#### Scenario: 使用配置启动 LiDAR 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `lidar` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: LiDAR 默认实验配置
项目 MUST 提供 LiDAR-only strong、lightweight、supervised 和包含 LiDAR 的 fusion 示例配置。所有默认 LiDAR primary 配置 MUST 使用当前 modular BEV encoder 默认参数。

#### Scenario: LiDAR 默认配置可构建
- **WHEN** 开发者加载 `configs/lidar/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、optimizer 和 scheduler
- **AND** 配置 MUST 使用 `model.primary`

#### Scenario: LiDAR fusion 示例配置可构建
- **WHEN** 开发者加载包含 LiDAR 的 `configs/fusion/*.yaml`
- **THEN** 系统 MUST 能构建对应 fusion primary model
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
项目 MUST 为每个受支持单模态 `image`、`radar`、`gps`、`lidar` 和 `mmwave` 提供统一命名的 canonical 配置矩阵。每个单模态目录 MUST 包含 `strong.yaml`、`lightweight.yaml` 和 `supervised.yaml`。canonical 配置 MUST 使用 `model.primary`、统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和输出目录语义。

#### Scenario: 单模态 strong 配置
- **WHEN** 开发者加载 `configs/<modality>/strong.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_strong`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_strong`

#### Scenario: 单模态 lightweight 配置
- **WHEN** 开发者加载 `configs/<modality>/lightweight.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_lightweight`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_lightweight`

#### Scenario: 单模态 supervised 配置
- **WHEN** 开发者加载 `configs/<modality>/supervised.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 不包含 `distillation`
- **AND** 配置 MUST 将被训练主模型配置为明确的 supervised baseline

#### Scenario: 旧单模态 KD 配置被拒绝
- **WHEN** 开发者加载 `configs/<modality>/logits_kd.yaml` 或 `configs/<modality>/rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向 strong、lightweight 或 supervised 入口

### Requirement: 单模态 legacy no-KD 入口兼容
项目 MUST 拒绝现有 `configs/<modality>/no_kd.yaml` 旧入口，并 MUST 在文档中说明其历史语义和推荐替代入口。

#### Scenario: image legacy no-KD 保持 student baseline
- **WHEN** 用户运行已退役的 `configs/image/no_kd.yaml`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 文档 MUST 引导新实验优先使用 `configs/image/lightweight.yaml` 或 `configs/image/supervised.yaml`

#### Scenario: radar GPS LiDAR legacy no-KD 保持 teacher baseline
- **WHEN** 用户运行已退役的 `configs/radar/no_kd.yaml`、`configs/gps/no_kd.yaml` 或 `configs/lidar/no_kd.yaml`
- **THEN** 系统 MUST 拒绝这些配置
- **AND** 文档 MUST 引导新实验优先使用对应 `strong.yaml`、`lightweight.yaml` 或 `supervised.yaml`

### Requirement: primary 角色不得受原脚本残留影响
配置驱动流程 MUST 以 YAML 中的 `model.primary` 作为被训练主模型。默认 canonical lightweight baseline MUST 使用 lightweight 注册名，不得默认使用旧 teacher-as-student 残留。

#### Scenario: no-KD 只训练配置中的主模型
- **WHEN** 配置使用当前 supervised/adaptation 入口
- **THEN** 训练流程 MUST 不构建或加载 frozen teacher
- **AND** optimizer MUST 只更新 `model.primary` 构建出的主模型

#### Scenario: canonical student baseline 使用 lightweight student
- **WHEN** 开发者加载任意 canonical `lightweight.yaml`
- **THEN** `model.primary.type` MUST 为对应 lightweight 注册名
- **AND** `model.primary.type` MUST NOT 等于对应 strong 注册名

#### Scenario: canonical KD 路径被拒绝
- **WHEN** 开发者加载任意 canonical `logits_kd.yaml` 或 `rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不构建 frozen teacher 或 distiller

### Requirement: canonical 配置命名与输出目录一致
canonical 配置 MUST 使用可预测的实验名和 run name。默认路径 MUST 便于用户按 strong/lightweight/supervised 或当前 workflow 顺序运行实验，并 MUST 支持命令行覆盖。

#### Scenario: canonical run name 与文件语义一致
- **WHEN** 开发者加载任意 canonical 配置
- **THEN** `experiment.name` MUST 与不含 `.yaml` 的文件 stem 一致
- **AND** `output.run_name` MUST 与 `experiment.name` 一致

#### Scenario: canonical 配置不解析 teacher checkpoint
- **WHEN** 用户加载当前 canonical 配置
- **THEN** 系统 MUST 不解析 teacher checkpoint
- **AND** 训练流程 MUST 只构建 primary model

#### Scenario: canonical KD checkpoint override 被拒绝
- **WHEN** 用户通过命令行覆盖 `distillation.teacher_model_name`
- **THEN** 配置加载 MUST 失败
- **AND** 错误信息 MUST 指向当前配置入口

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
默认评估工作流 MUST 不依赖固定 `outputs/<run_name>/checkpoints/best.pth` 作为唯一权重来源。当固定 `run_name` 已存在导致新训练输出目录追加时间戳时，后续评估 MAY 通过 registry 找到对应配置的最高验证 Top-1 checkpoint。

#### Scenario: 时间戳输出被评估复用
- **WHEN** 训练因为目标运行目录已存在而写入带时间戳后缀的新运行目录
- **THEN** 训练完成后 registry MAY 保存该运行的最高验证 Top-1 checkpoint
- **AND** 后续评估 MUST 能显式指定或从 registry 解析该 checkpoint

#### Scenario: 旧 KD 权重路径不再作为训练 fallback
- **WHEN** 用户已有旧式 `paths.weights_dir / teacher_model_name` checkpoint 且 registry 没有匹配候选
- **THEN** 当前训练流程 MUST 不使用该路径加载 teacher
- **AND** 评估入口仍可通过 `--weights` 显式指定待评估 checkpoint

### Requirement: mmWave 配置驱动实验
项目 MUST 支持通过配置文件启动 mmWave-only 训练和评估。mmWave-only 配置 MUST 使用 `experiment.task: mmwave`、当前 mmWave dataset contract 和 `model.primary`，并通过统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和指标流程运行。

#### Scenario: 使用配置启动 mmWave-only 训练
- **WHEN** 用户通过当前 CLI 传入 mmWave-only 训练配置
- **THEN** 系统 MUST 构建包含 mmWave 输入的 dataset、配置指定的 mmWave primary model、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 不要求图像、雷达、GPS、LiDAR、teacher checkpoint 或 distiller
- **AND** mmWave 输入 MUST 使用 `[B, T, 64]` 的 dB receive-power 特征序列

#### Scenario: 使用配置启动 mmWave-only 评估
- **WHEN** 用户通过当前 CLI 传入 mmWave-only 评估配置和 mmWave 模型权重
- **THEN** 系统 MUST 构建配置指定的 mmWave 模型并只使用 mmWave 输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标
- **AND** 评估流程 MUST 复用训练时保存的 mmWave scaler

### Requirement: mmWave fusion 配置驱动实验
项目 MUST 支持通过 fusion `modalities` 配置启用 mmWave。包含 mmWave 的 fusion 配置 MUST 复用统一 fusion 训练和评估流程，并 MUST 构建单个 fusion primary model。

#### Scenario: 使用配置启动五模态 fusion 训练
- **WHEN** 用户通过训练入口传入 `modalities: ["image", "radar", "gps", "lidar", "mmwave"]` 的 fusion 配置
- **THEN** 系统 MUST 构建五个模态输入所需的 dataset 字段和 fusion primary model
- **AND** 系统 MUST 在 batch 准备阶段构造 image、radar、gps、lidar 和 mmWave 输入

#### Scenario: 使用配置启动 mmWave 参与的双模态 fusion 训练
- **WHEN** 用户通过训练入口传入包含 `mmwave` 的任意合法双模态 fusion 配置
- **THEN** 系统 MUST 只准备 `modalities` 中列出的模态输入
- **AND** 未启用的模态字段 MUST 不影响训练启动

### Requirement: mmWave 默认实验配置
项目 MUST 提供 mmWave-only strong、lightweight、supervised 配置和包含 mmWave 的 canonical fusion 配置。所有默认 mmWave primary 配置 MUST 使用 `mmwave_input_size: 64`、`mmwave_normalize: true` 和 `gru_params: [64, 64, 1]`。

#### Scenario: mmWave 默认配置可构建
- **WHEN** 开发者加载 `configs/mmwave/*.yaml`
- **THEN** 系统 MUST 能构建对应 dataset、model、loss、optimizer 和 scheduler
- **AND** primary 配置的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** primary 配置的 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave KD 配置被拒绝
- **WHEN** 用户运行 `configs/mmwave/logits_kd.yaml` 或 `configs/mmwave/rkd.yaml`
- **THEN** 配置加载 MUST 失败
- **AND** 系统 MUST 不解析 teacher checkpoint

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
训练和默认评估流程 MUST 按 DeepSense6G scene 或 scenegroup 归类输出运行目录。默认输出根目录保持 `outputs`，单场景 DeepSense6G 运行目录 MUST 写入 `outputs/<scene_slug>/<run_name>/` 或等价的用户配置根目录下；多场景 DeepSense6G 运行目录 MUST 写入 `outputs/scenegroup_<scene-range-or-list>/<run_name>/` 或等价的用户配置根目录下。评估矩阵和成组评估输出 MUST 优先写入 `outputs/evaluations/<study_id>/`，除非用户显式传入完整输出目录。

#### Scenario: 显式 Scenario 9 训练输出归档到 scene9
- **WHEN** 用户显式选择 Scenario 9 并运行训练且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scene9/<run_name>/`
- **AND** checkpoints、`final_config.yaml`、`train_log.json`、metrics、TensorBoard event 和训练曲线 MUST 都写入该运行目录

#### Scenario: 默认 Scenario 31 训练输出归档到 scene31
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 单场景训练配置
- **THEN** 运行目录 MUST 创建在 `outputs/scene31/<run_name>/`
- **AND** 同名 Scenario 9、Scenario 32 或 scenegroup 运行目录不得被覆盖

#### Scenario: 显式 Scenario 32 训练输出归档到 scene32
- **WHEN** 用户显式选择 Scenario 32 并运行训练且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scene32/<run_name>/`
- **AND** 同名 Scenario 31 或 scenegroup 运行目录不得被覆盖

#### Scenario: 多场景训练输出归档到 scenegroup
- **WHEN** 用户运行包含多个 DeepSense6G scene 的训练配置且 `output.dir: outputs`
- **THEN** 运行目录 MUST 创建在 `outputs/scenegroup_<scene-range-or-list>/<run_name>/`
- **AND** 同名单场景运行目录不得被覆盖

#### Scenario: resume 使用默认场景或 scenegroup 运行目录
- **WHEN** 用户设置 `training.resume: true`、固定 `output.run_name` 且使用默认输出根
- **THEN** 系统 MUST 从当前配置对应的 `outputs/<scene-or-scenegroup>/<run_name>/checkpoints/last.pth` 恢复训练
- **AND** 系统不得回退到不同 scene 或 scenegroup 的同名运行目录

#### Scenario: 显式评估输出目录保持完整路径
- **WHEN** 用户通过评估入口显式传入 `--output-dir`
- **THEN** 系统 MUST 使用该目录作为完整输出目录
- **AND** 系统不得额外追加 `scene_slug` 或 scenegroup slug

### Requirement: 场景选择命令行覆盖
训练和评估入口 MUST 支持通过现有 dotted override 选择场景，不需要新增独立 CLI 参数。

#### Scenario: 命令行覆盖到 Scenario 9
- **WHEN** 用户运行 `kd-sensing-train --config <config> data.dataset.scene=9`
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
- **WHEN** 用户运行 `kd-sensing-train --config configs/fusion/gps_mmwave_lightweight.yaml`
- **THEN** 系统 MUST 解析该 canonical path 并启动 fusion lightweight 训练流程
- **AND** 训练流程 MUST 不要求 `configs/fusion/gps_mmwave_lightweight.yaml` 在磁盘上存在

#### Scenario: 评估入口使用虚拟 canonical 配置
- **WHEN** 用户运行 `kd-sensing-evaluate --config configs/fusion/gps_mmwave_lightweight.yaml --weights <path>`
- **THEN** 系统 MUST 解析该 canonical path 并构建对应 fusion primary 模型
- **AND** 评估流程 MUST 只准备该配置启用的模态输入

#### Scenario: dry-run 使用虚拟 canonical 配置
- **WHEN** 用户运行 `kd-sensing-train --config configs/fusion/gps_mmwave_lightweight.yaml --dry-run`
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
训练和评估流程 MUST 在运行产物中记录足够的 split 协议信息，用于判断不同实验是否使用同一数据协议并可横向比较。记录 MUST 包含实际 CSV 路径、样本数和 split metadata 路径或核心字段。对于 MMW Town10 或其它滑窗 sequence 数据，记录还 MUST 包含 `split_strategy`、`split_protocol_version`、`strict_validation_eligible`、`eligibility_reasons` 和可用的 leakage diagnostics 摘要，避免把 unknown 或高重叠 split 误当成 strict validation 结果。

#### Scenario: 训练输出包含 split metadata 引用
- **WHEN** 训练入口构建 train/test dataset
- **THEN** `final_config.yaml`、`train_log.json` 或等价运行产物 MUST 记录 split metadata 路径或核心字段
- **AND** 记录 MUST 包含 split 策略、seed、train/test `seq_index` 数量和 train/test 样本数
- **AND** 当 split metadata 包含 strict eligibility 或 leakage diagnostics 时，运行产物 MUST 记录这些字段

#### Scenario: 评估输出包含 split 协议
- **WHEN** 评估入口构建 test dataset
- **THEN** 评估报告 MUST 记录实际使用的 test CSV 和可用的 split 协议信息
- **AND** 当当前 CSV 缺少 split metadata 时，系统 MUST 给出清晰错误或显式警告，避免把未知 split 协议误当成新协议结果
- **AND** 当 split metadata 标记 `strict_validation_eligible=false` 时，评估报告 MUST 保留指标但标记其不适合作为 strict 主结论

#### Scenario: 跨模态 split 可比较
- **WHEN** 用户使用同一组 train/test CSV 运行 image、radar、GPS、LiDAR、mmWave 或 fusion 实验
- **THEN** 各运行产物中的 split 协议信息 MUST 能显示它们使用相同 CSV 和相同 split metadata
- **AND** 如果 CSV 路径、split metadata、split strategy 或 strict eligibility 不同，用户 MUST 能从运行产物中看出这些结果不应直接作为同一 split 协议比较

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
- **AND** 该 helper 的语义 MUST 表示长时序输出对齐，不得作为方法专属额外 prediction slot 的兼容承诺

#### Scenario: 输出 slot 不足时报错
- **WHEN** 模型输出 logits 的时间维长度小于 `num_pred`
- **THEN** 训练、验证或评估流程 MUST 报出清晰错误
- **AND** 系统 MUST 不通过重复、padding 或拼接历史 beam 自动补齐 prediction slots

#### Scenario: 诊断预测导出保留 t+1
- **WHEN** viewer prediction export 写出 `confidence_curves` 或 `beam_distribution`
- **THEN** 导出的第一个 horizon MUST 表示 `t+1`
- **AND** 导出逻辑 MUST 不把第一个预测 slot 当作 current beam 丢弃

### Requirement: Future horizon flat metrics
验证和评估输出 MUST 在现有 nested top-k 数组之外，增加 future horizon 扁平指标字段。字段 MUST 使用 `t1/t2/t3/avg` 命名，并 MUST 不输出历史 current beam 或 h0 指标。

#### Scenario: 保存三步 Top-K 扁平字段
- **WHEN** 验证阶段产出 logits `[B,3,64]` 和 labels `[B,3]`
- **THEN** `metrics.json` MUST 包含 `val_top1_t1`、`val_top1_t2`、`val_top1_t3` 和 `val_top1_avg`
- **AND** `metrics.json` MUST 包含 `val_top3_avg` 和 `val_top5_avg`
- **AND** 这些 avg 字段 MUST 对有效 future horizon 求平均

#### Scenario: 不输出旧 h0 指标
- **WHEN** 普通 future-only 评估写出 metrics
- **THEN** metrics MUST 不包含 `top1_h0`
- **AND** metrics MUST 不包含 `top1_future_avg`
- **AND** metrics MUST 不包含 `beam8_acc`

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

### Requirement: 共享 evaluation pass
训练验证、force-mask subset 验证和 standalone evaluate MUST 复用同一个 evaluation pass 完成 batch 准备、model forward、objective loss、输出收集、指标聚合和 available metrics 生成。各入口 MAY 对结果做输出包装或文件写出，但 MUST 不复制核心 forward/loss/collect 逻辑。

#### Scenario: 普通验证使用共享 pass
- **WHEN** 训练流程在 epoch 结束后调用 validation
- **THEN** validation MUST 通过共享 evaluation pass 计算 loss、Top-K、DBA 和 objective 指标
- **AND** 返回的公开 metrics 键 MUST 保持与变更前兼容

#### Scenario: force-mask subset 使用共享 pass
- **WHEN** evaluation 配置启用 modality subset 或 force mask 验证
- **THEN** subset validation MUST 使用同一个 evaluation pass 并传入 mask 选项
- **AND** subset 结果 MUST 包含与普通验证一致的 objective metadata 和 available metrics

#### Scenario: standalone evaluate 使用共享 pass
- **WHEN** 用户通过评估入口运行 checkpoint evaluate
- **THEN** evaluate MUST 使用共享 evaluation pass 计算指标
- **AND** 保存的报告 MUST 与训练验证使用同一套 objective 指标语义

### Requirement: 评估指标写出与 runtime metadata 对齐
训练和评估写出的 metrics/report MUST 包含 objective runtime metadata、primary metric、available metrics 和已启用模态信息。该 metadata MUST 来自 objective 与 modality resolution 层，而不是入口各自手写推导。

#### Scenario: 评估报告记录 objective metadata
- **WHEN** 用户评估 `experiment.objective: occlusion` 的模型
- **THEN** 评估报告 MUST 记录 objective 名称、primary loss、primary metric、metric mode、enabled targets 和 enabled heads
- **AND** 这些字段 MUST 与训练 final config 中的 prediction objective metadata 一致

#### Scenario: 评估报告记录启用模态
- **WHEN** 用户评估 GPS+mmWave fusion 模型
- **THEN** 评估报告 MUST 记录启用模态为 `["gps", "mmwave"]`
- **AND** 该模态集合 MUST 由统一模态解析逻辑产生

### Requirement: Snapshot workflow metadata
训练、验证和评估流程 MUST 在 snapshot next-frame baseline 的运行产物中记录该实验的无历史窗口语义。metadata MUST 足以让结果汇总工具区分 snapshot baseline 与历史窗口 baseline。

#### Scenario: 训练记录 snapshot metadata
- **WHEN** 用户训练 snapshot next-frame baseline
- **THEN** 运行 metadata MUST 记录 `variant: snapshot_next_frame`
- **AND** MUST 记录 `seq_len: 1` 和 `num_pred: 1`
- **AND** MUST 记录 `uses_history_window: false`
- **AND** MUST 记录 `uses_temporal_core: false`

#### Scenario: 评估报告记录 snapshot metadata
- **WHEN** 用户评估 snapshot next-frame baseline checkpoint
- **THEN** 评估报告 MUST 包含 checkpoint 或配置中的 snapshot metadata
- **AND** 报告 MUST 记录 enabled modalities、objective、scene、train/validation split CSV 和样本数
- **AND** 报告 MUST 标记 validation split 是 80/20 协议中的验证集合

### Requirement: Snapshot smoke workflow
项目 MUST 提供可通过统一训练入口运行的 snapshot smoke workflow。该 workflow MUST 使用 `conda run -n kd_mm_beam` 运行测试、训练或评估命令。

#### Scenario: 单模态 snapshot smoke test
- **WHEN** 开发者运行单模态 snapshot 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、validation 和 checkpoint 保存
- **AND** 日志中的模型配置 MUST 显示无 GRU representation core

#### Scenario: 多模态 snapshot smoke test
- **WHEN** 开发者运行五模态 snapshot fusion 配置的最小训练 smoke test
- **THEN** 训练流程 MUST 通过现有 fusion batch preparation 构造启用模态输入
- **AND** forward 输出 MUST 与 `num_pred=1` 的 labels 对齐
- **AND** 训练流程 MUST 不加载 teacher checkpoint

### Requirement: Snapshot 与历史窗口比较输出
实验工作流 MUST 允许用户在同一 Scenario 31 和同一 objective 下比较 snapshot baseline 与历史窗口 baseline。比较输出 MUST 明确展示实验变体和 split 协议，避免把不同时间上下文或不同窗口生成口径的结果混为同一条件。

#### Scenario: 记录 split 协议差异
- **WHEN** 用户对同一模态运行 snapshot baseline 和历史窗口 baseline
- **THEN** 两次运行的 metadata MUST 记录各自 train/validation CSV 路径和样本数
- **AND** 如果 CSV 路径或样本数不同，比较工具或文档 MUST 要求用户将其视为不同数据口径

#### Scenario: 结果表包含时间上下文
- **WHEN** 工具汇总 snapshot 与历史窗口结果
- **THEN** 表格或 JSON 输出 MUST 包含 `variant`、`seq_len`、`num_pred`、`uses_temporal_core` 和 `split_protocol`
- **AND** 模态强弱排序 MUST 能按这些字段分组计算

### Requirement: Resolved config artifact and startup summary
Every debug run MUST save the fully resolved configuration and print a startup summary of the fields needed to compare experiment variants. The summary MUST be generated after defaults, aliases and command-line overrides are applied.

#### Scenario: 保存 resolved config
- **WHEN** a debug run starts
- **THEN** the run output directory MUST contain `resolved_config.yaml` or an equivalent fully resolved config artifact
- **AND** the artifact MUST reflect defaults, generated config values, aliases and command-line overrides

#### Scenario: 打印关键配置摘要
- **WHEN** a debug run starts
- **THEN** startup logs MUST include modalities, dataset path, train/val split paths, `seq_len`, `num_pred`, `num_classes`, batch size, optimizer, learning rate, scheduler and max epochs
- **AND** startup logs MUST include model type, CSI encoder type, `d_model`, `delay_taps`, `view_fusion`, `use_internal_gru`, pilot estimator enabled/mode/SNR, `csi_hardening.enabled` and `csi_degradation.enabled`

### Requirement: Baseline clone config diff artifact
The experiment workflow MUST support comparing a generated baseline clone against a reference baseline resolved config. The diff MUST separate allowed run identity differences from key behavior differences.

#### Scenario: 生成 A0 clone diff
- **WHEN** both `A0_original` and `A0_clone_generated` resolved configs are available
- **THEN** the workflow MUST produce a diff artifact comparing them
- **AND** the diff MUST ignore only allowlisted run identity fields such as run name, output directory, timestamp and seed when configured

#### Scenario: 关键字段差异失败
- **WHEN** the diff finds a difference in optimizer, scheduler, loss, dataset split, normalization, train RMS path, `seq_len`, `num_pred`, `num_classes`, model type, CSI encoder, representation core or beam head
- **THEN** the workflow MUST mark the parity check as failed
- **AND** the failure message MUST list the differing config paths

### Requirement: Module trainability startup report
The training workflow MUST report trainable parameter counts by major module for debug runs. The report MUST distinguish CSI encoder, representation core, beam head and fusion modules when those modules exist.

#### Scenario: 打印模块参数统计
- **WHEN** a debug run builds the model
- **THEN** startup logs MUST include total parameter count and total trainable parameter count
- **AND** startup logs MUST include trainable parameter counts by CSI encoder, representation core, beam head and fusion module where present

#### Scenario: 发现模块无可训练参数
- **WHEN** a required trainable module has zero trainable parameters
- **THEN** startup logs MUST mark the module as suspicious
- **AND** the warning MUST include the module name and resolved model path

### Requirement: Debug metrics logging
The training workflow MUST persist debug diagnostics in machine-readable run logs when debug mode is enabled. The diagnostics MUST be scoped so normal runs are unaffected when debug mode is disabled.

#### Scenario: 持久化首 batch 诊断
- **WHEN** CSI first-batch debug diagnostics are produced
- **THEN** the workflow MUST write them to the run log, metadata artifact or TensorBoard text/scalar stream
- **AND** the stored record MUST distinguish train and validation batch sources

#### Scenario: 持久化 epoch 训练健康指标
- **WHEN** epoch-level grad norm and param delta diagnostics are produced
- **THEN** the workflow MUST append them to the epoch metrics log
- **AND** normal training metrics arrays MUST remain backward compatible for existing analysis scripts

### Requirement: CSI hardening sweep validity gate
CSI hardening sweep 的分析流程 MUST 在候选排序和设计结论前执行有效性 gate。有效性 gate MUST 至少检查 A0 clone parity、pilot 噪声量级、C1/C2 单变量健康状态和必需 diagnostics 是否存在。未通过 gate 的 sweep MUST 标记为 invalid 或 pending-debug，不得被解释为 hardening 设计失败。

#### Scenario: A0 parity 未通过
- **WHEN** `A0_clone_generated` 未通过与 `A0_original` 的关键配置 diff 或短跑曲线 parity
- **THEN** 分析输出 MUST 将 full sweep 状态标记为 pending-debug 或 invalid
- **AND** 系统 MUST 不输出 slow-high-ceiling 候选结论

#### Scenario: pilot 噪声量级失真
- **WHEN** 一个标记为 mild pilot estimation 的 run 的 `noise_power_signal_ratio` 明显高于其配置 SNR 对应范围
- **THEN** 分析输出 MUST 将该 run 标记为 `invalid_due_to_pilot_noise_scale` 或等价原因
- **AND** 该 run MUST 不参与 slow-high-ceiling 候选排序

#### Scenario: C1 或 C2 单变量异常
- **WHEN** A0 clone 正常学习但 C1 view gate warmup only 或 C2 no internal GRU only 掉到接近随机水平
- **THEN** 分析输出 MUST 将问题归因到对应 encoder 单变量路径
- **AND** 系统 MUST 阻止把 B/D hardening 组合结果解释为 hardening 强度问题

#### Scenario: 旧 sweep 缺少必需 diagnostics
- **WHEN** 分析脚本处理旧 full sweep 目录且该目录缺少 A0 parity、pilot noise ratio 或 debug decision artifacts
- **THEN** 分析输出 MUST 明确标记该 sweep 需要重跑或人工确认
- **AND** 默认候选排序 MUST 排除这些 invalid/pending run

### Requirement: CSI hardening sweep rerun workflow
项目 MUST 提供修复后的 CSI-only A/B/C/D sweep 运行入口或命令说明。该 workflow MUST 先运行短 debug gate，再运行完整 CSI-only sweep，并在输出中记录所使用的配置版本、pilot estimation 模式、noise ratio diagnostics 和旧结果隔离状态。

#### Scenario: 生成修复后的 A1 配置
- **WHEN** 开发者生成或加载修复后的 A1 mild pilot estimation 配置
- **THEN** 配置 MUST 使用 estimation-SNR 模式
- **AND** resolved config MUST 记录固定 SNR 或训练 SNR 采样区间

#### Scenario: 生成修复后的 B/C/D 配置
- **WHEN** 开发者生成或加载修复后的 B、C 或 D 组配置
- **THEN** 每个配置 MUST 显式关闭 pilot estimation noise
- **AND** 每个配置 MUST 保留自身声明的 hardening 或 encoder 变量

#### Scenario: 重跑前执行 debug gate
- **WHEN** 开发者请求完整 CSI hardening sweep
- **THEN** workflow MUST 先确认 A0 original、A0 clone、pilot disabled、C1 only 和 C2 only 的 debug gate 通过
- **AND** 如果 gate 未通过，workflow MUST 停止或将完整 sweep 输出标记为 pending-debug

#### Scenario: 输出新旧结果隔离状态
- **WHEN** 修复后的 sweep analysis 完成
- **THEN** summary artifact MUST 记录当前 sweep 是否基于修复后的 pilot scaling 配置
- **AND** 如果同一项目中存在旧 invalid sweep，summary artifact MUST 不把旧 sweep 的候选结果混入当前 ranking

### Requirement: Raymobtime s008 workflow 已退役
Raymobtime s008 预处理、训练、评估、smoke 和实验矩阵 workflow 已退役，不属于当前实验入口。旧 `configs/raymobtime/*`、`configs/preprocess/raymobtime_s008_*.yaml`、`raymobtime_s008` dataset/model/preprocessor 名称和 selection 模型名称 MUST 只作为 migration guard 命中或历史说明出现。

#### Scenario: 旧 Raymobtime 预处理配置被拒绝
- **WHEN** 用户运行 `kd-sensing-preprocess` 并引用 Raymobtime s008 预处理配置或 preprocessor type
- **THEN** 系统 MUST fail fast
- **AND** 错误信息 MUST 明确 Raymobtime s008 已退役且无兼容迁移入口

#### Scenario: 旧 Raymobtime 训练配置被拒绝
- **WHEN** 用户运行退役历史命令 `conda run -n kd_mm_beam kd-sensing-train --config configs/raymobtime/s008_multitask_selection.yaml`
- **THEN** 系统 MUST 不构建 `raymobtime_s008` dataset、selection 模型或 Raymobtime cache
- **AND** 错误信息 MUST 指向当前保留 workflow 或说明该研究线已退役

#### Scenario: Raymobtime smoke 与矩阵不作为当前要求
- **WHEN** 开发者运行当前架构边界、config load 或实验 workflow 测试
- **THEN** 测试 MUST 不要求 Raymobtime dataset smoke、训练 smoke、评估 smoke 或推荐实验矩阵存在
- **AND** 若测试覆盖 Raymobtime 名称，MUST 只验证 migration guard 或 registry 拒绝语义

### Requirement: 训练编排重构保持输出兼容
训练编排内部重构后，训练入口 MUST 保持现有用户可见输出和恢复语义兼容。`final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`training_outputs.npz`、checkpoint、checkpoint sidecar、teacher metrics、TensorBoard events 和 debug artifacts 的关键字段、路径和含义 MUST 与变更前兼容，除非对应 change 明确声明 breaking change。

#### Scenario: 训练日志字段兼容
- **WHEN** 开发者运行 synthetic 或 fixture 短训练并完成至少一个 epoch
- **THEN** `train_log.json` MUST 包含历史兼容的 history 字段、`epoch_logs`、`early_stopping`、`runtime`、`prediction_objective`、`normalization_artifacts` 和 `checkpoint_loads`
- **AND** active objective 的指标字段 MUST 与 objective metadata 声明一致

#### Scenario: training_outputs npz 兼容
- **WHEN** 训练完成并写出 `training_outputs.npz`
- **THEN** 该文件 MUST 包含现有分析脚本依赖的 history 数组、objective 名称、primary loss、primary metric、enabled targets、enabled heads 和 loss weights
- **AND** inactive optional metrics MUST 使用既有 null/NaN 兼容语义表示不可用

#### Scenario: checkpoint metadata 兼容
- **WHEN** 训练保存 `best.pth`、`best_top1.pth` 或 `last.pth`
- **THEN** checkpoint 和 sidecar MUST 继续记录 selection metric、selection mode、selected epoch、objective metric、task metrics、split metadata 和 normalization artifacts
- **AND** 恢复训练 MUST 继续兼容缺少通用 early stopping metadata 的历史 checkpoint

#### Scenario: TensorBoard tag 兼容
- **WHEN** 训练启用 TensorBoard
- **THEN** 系统 MUST 继续写入当前 objective 对应的 TensorBoard scalar tag
- **AND** 用户显式启用 legacy accuracy tags 时，历史 `accuracy/*` 和 `dba/val_adba` tag MUST 继续可选写入

### Requirement: 训练配置重构提供 characterization 检查
项目 MUST 为训练编排和配置加载重构提供快速 characterization 检查，覆盖关键输出契约、config load 顺序、CLI help 和架构边界。检查 MUST 使用 `kd_mm_beam` 环境，并 MUST 不依赖真实数据、长时间训练或新生成 checkpoint 纳入源码。

#### Scenario: 训练短流程 characterization
- **WHEN** 开发者运行本变更记录的训练短流程测试
- **THEN** 测试 MUST 完成 forward、loss、backward、validation、checkpoint 和 artifact 写出
- **AND** 测试 MUST 验证重构后的关键输出字段与兼容契约一致

#### Scenario: config load characterization
- **WHEN** 开发者运行 config loading focused tests
- **THEN** 测试 MUST 覆盖实体 YAML、virtual canonical 配置、snapshot 配置、Raymobtime migration guard 和命令行覆盖
- **AND** 测试 MUST 验证 normalization 与 validation 结果保持兼容

#### Scenario: CLI help characterization
- **WHEN** 开发者运行 CLI help focused tests
- **THEN** `kd-sensing-train --help`、`kd-sensing-evaluate --help`、`kd-sensing-preprocess --help`、`kd-sensing-jepa-visual-analysis --help` 和 `kd-sensing-jepa-gps-shortcut-benchmark --help` MUST 正常退出
- **AND** 检查 MUST 不读取真实数据集、不加载 checkpoint、不启动训练

### Requirement: 推荐实验文档保持精简入口
实验工作流文档 MUST 将 README 作为入口地图，而不是完整实验手册。README MUST 指向 canonical config、docs 和 OpenSpec；详细实验矩阵、分析流程和调参说明 MUST 放在 `docs/` 或对应 specs 中。已退役的 G2D、CRAF、MARF 和 Multimodal-NF 内容 MUST 从 README 推荐入口和实验矩阵中删除。

#### Scenario: README 提供最短可运行路径
- **WHEN** 新用户阅读 README
- **THEN** 用户 MUST 能找到安装命令、快速健康检查、训练/评估/预处理/manifest 导出入口和数据产物边界
- **AND** 用户 MUST 能通过链接进入当前保留能力的详细实验矩阵或 viewer 文档

#### Scenario: 长实验说明迁移到 docs
- **WHEN** README 中的某段内容主要描述当前保留的 CSI hardening、MMW、JEPA 或诊断 benchmark 详细实验流程
- **THEN** 该内容 MUST 迁移到对应 `docs/` 文件或 OpenSpec spec
- **AND** README MUST 保留简短摘要和链接

#### Scenario: 退役研究线文档删除
- **WHEN** README、docs 或实验矩阵提到 G2D、CRAF、MARF 或 Multimodal-NF 推荐运行命令
- **THEN** 这些段落 MUST 被删除或改为明确说明该入口已退役
- **AND** 文档 MUST 不再推荐运行对应配置、测试或日志分析流程

### Requirement: 表面积收敛保持实验 artifact 兼容
删除冗余配置、入口或文档后，当前保留的训练和评估 workflow MUST 继续保存完整运行 artifact。使用保留的 virtual/overlay 配置时，运行目录 MUST 记录足够信息用于复现，不得要求用户恢复已删除的实体 YAML。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 配置不得由 virtual alias 接管。

#### Scenario: virtual 配置训练 artifact 完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置启动训练并完成 artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、`train_log.json`、checkpoint metadata 和 split/runtime metadata
- **AND** 这些 artifact MUST 足以说明实际模型、数据、loss、训练参数和 checkpoint 来源

#### Scenario: 删除 fallback 入口不影响 console script
- **WHEN** 重复脚本 wrapper 被删除
- **THEN** 对应 console script 或 `python -m kd_sensing.cli.*` 入口 MUST 继续通过 `--help` 检查
- **AND** README 推荐命令 MUST 使用仍存在的入口

#### Scenario: 研究脚本不进入核心 workflow 兼容承诺
- **WHEN** 保留的研究脚本未声明为包内 CLI
- **THEN** 核心训练、评估、预处理和 manifest 导出 workflow MUST 不依赖该脚本
- **AND** 该脚本的输出产物 MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录

#### Scenario: 退役配置不被兼容接管
- **WHEN** 用户引用已删除的 CRAF、MARF、G2D 或 Multimodal-NF 配置路径
- **THEN** 配置加载器 MUST 给出清晰缺失或退役错误
- **AND** 系统 MUST 不生成同名 virtual 配置

### Requirement: 源码表面积优化必须保持核心 workflow 兼容
删除冗余配置、拆分源码模块或收敛入口后，训练、评估、预处理和当前研究诊断的公开工作流 MUST 保持现有用户可见语义。实现 MAY 调整内部模块位置，但 MUST 不要求用户改用未记录的新命令。已退役的 viewer manifest 导出不属于该兼容承诺。

#### Scenario: 核心 CLI help 继续可用
- **WHEN** 本 change 完成后开发者运行核心入口 help 检查
- **THEN** `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-jepa-visual-analysis` 和 `kd-sensing-jepa-gps-shortcut-benchmark` MUST 正常退出
- **AND** 对应包内 CLI 模块 MUST 继续可通过 `python -m kd_sensing.cli.<name> --help` 调用

#### Scenario: 拆分模块不改变公共返回结构
- **WHEN** 用户通过既有公开函数或 CLI 运行训练、评估、预处理或当前研究诊断
- **THEN** 返回 payload、日志字段、诊断字段和主要输出路径语义 MUST 与拆分前兼容
- **AND** 内部模块重命名 MUST 不要求用户修改配置文件中的公共字段

### Requirement: 删除实体配置后 workflow 必须可复现
当当前保留的实体 YAML 被 recipe/overlay 替代后，训练和评估 workflow MUST 继续保存足够的 resolved/final 配置、运行元数据和 checkpoint 来源信息，保证不恢复被删除 YAML 也能理解实际运行参数。已退役的 CRAF、MARF、G2D 和 Multimodal-NF 实体 YAML 删除后 MUST 不提供同名 recipe/overlay 兼容。

#### Scenario: virtual 配置训练记录完整
- **WHEN** 用户使用当前保留的 virtual/overlay 配置完成训练或 dry-run artifact 写出
- **THEN** 运行目录 MUST 包含完整 `final_config.yaml`、`resolved_config.yaml`、训练元数据和 checkpoint 来源信息
- **AND** 这些 artifact MUST 能说明实际模型、数据、loss、训练参数和输出 run name

#### Scenario: 删除 YAML 不影响评估入口
- **WHEN** 某个当前保留的实体 YAML 被删除但对应 virtual/overlay 配置仍被声明支持
- **THEN** `kd-sensing-evaluate --config <deleted-yaml-path>` MUST 通过配置加载器解析等价最终配置
- **AND** 如果该路径未被声明支持，系统 MUST 抛出清晰缺失配置错误

#### Scenario: 退役 YAML 不支持 virtual fallback
- **WHEN** 被删除 YAML 属于 CRAF、MARF、G2D 或 Multimodal-NF
- **THEN** 系统 MUST 将其视为不支持路径
- **AND** 系统 MUST 不为其提供 virtual fallback

### Requirement: 入口收敛不得让研究脚本成为核心依赖
保留的 `scripts/` 和 `tools/analysis/` 研究或支持脚本 MUST 不成为核心训练、评估、预处理或 manifest 导出 workflow 的必需依赖。仓库级 `tools/visualization/` viewer support 已退役，核心 workflow MUST 通过包内模块或 package console script 完成。

#### Scenario: 训练入口不依赖研究脚本
- **WHEN** 用户运行 `kd-sensing-train` 或 `python -m kd_sensing.cli.train`
- **THEN** 训练 workflow MUST 不要求调用 `scripts/analyze_*`、`tools/analysis/*` 或 viewer 支持脚本
- **AND** 研究脚本删除或重分类 MUST 不破坏核心训练入口

#### Scenario: viewer manifest 边界清晰
- **WHEN** 用户引用 viewer manifest 导出或 `kd-sensing-visualize-modalities` 兼容入口
- **THEN** 系统 MUST 拒绝该退役入口或不再提供该入口
- **AND** 仓库级 Gradio viewer entrypoint MUST 不再作为当前支持脚本保留

### Requirement: 本 change 不改变本地产物策略
源码、配置和入口优化完成后，本地产物策略 MUST 保持现状。工作流 MAY 继续生成 outputs、logs、cache 和 checkpoint，但本 change MUST 不要求清理、压缩、迁移或提交这些产物。

#### Scenario: 训练输出仍位于忽略路径
- **WHEN** 用户在本 change 后运行训练或评估并生成输出
- **THEN** 新的 logs、outputs、cache 和 checkpoint MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录
- **AND** 文档 MUST 不要求将这些本地产物加入源码变更

#### Scenario: 不要求清理已有产物
- **WHEN** 开发者实施本 change 的任务
- **THEN** 任务验收 MUST 不包含删除、压缩或迁移既有 `dataset/`、`outputs/`、`logs/` 文件
- **AND** 测试和 OpenSpec 校验 MUST 能在不修改这些本地产物的情况下完成

### Requirement: 运行状态产物
训练和评估入口 MUST 尽量写出机器可读运行状态产物，使 run index 能判断启动、正常完成和 Python 异常失败。状态产物 MUST 保持轻量，并且 MUST 不改变现有 `final_config.yaml`、`resolved_config.yaml`、`metrics.json`、`train_log.json`、checkpoint 或 TensorBoard 语义。

#### Scenario: 训练启动写出状态
- **WHEN** 训练入口创建 run_dir 并完成初始配置解析
- **THEN** 系统 MUST 写出 `run_status.json` 或等价 runtime status 字段
- **AND** 状态 MUST 至少包含 `state: running`、run_dir、config path、start time、pid、experiment name、task、objective 和 enabled modalities

#### Scenario: 训练正常完成更新状态
- **WHEN** 训练完成并写出最终 metrics、train log 和 checkpoint metadata
- **THEN** 系统 MUST 将运行状态更新为 `complete`
- **AND** 状态 MUST 记录 end time、duration、primary metric、best checkpoint 和 metrics path

#### Scenario: Python 异常失败更新状态
- **WHEN** 训练或评估入口捕获到未处理 Python exception 并准备退出
- **THEN** 系统 SHOULD 将运行状态更新为 `failed`
- **AND** 状态 SHOULD 记录异常类型、异常消息和可查看的日志路径

#### Scenario: SIGKILL 无法捕获
- **WHEN** 训练进程被系统或用户以不可捕获方式终止
- **THEN** 系统 MAY 无法更新运行状态产物
- **AND** run index MUST 仍能通过日志和 partial artifacts 推断 killed、stale 或 partial 状态

### Requirement: Artifact schema 拆分兼容
训练和评估相关模块拆分后，用户可见 artifact schema MUST 保持兼容。`final_config.yaml`、`resolved_config.yaml`、`train_log.json`、`training_outputs.npz`、`metrics.json`、checkpoint sidecar、teacher metrics 和 TensorBoard tag 的关键字段、路径和含义 MUST 不因内部模块移动而改变。

#### Scenario: 训练 artifact 字段保持
- **WHEN** 训练流程内部 writer、objective metadata 或 runtime metadata helper 被拆分
- **THEN** `final_config.yaml`、`train_log.json` 和 `metrics.json` 中既有公开字段 MUST 保持可用
- **AND** focused tests MUST 覆盖关键字段 presence

#### Scenario: objective metadata 拆分后兼容
- **WHEN** objective metadata 表、alias、history fields 或 TensorBoard schema 被迁移到窄模块
- **THEN** 训练、验证和评估 MUST 继续解析同一组 objective、metric alias、metric mode 和 history fields
- **AND** 现有 objective tests MUST 保持通过

### Requirement: Metric horizon aggregation consistency
训练验证、force-mask subset 验证和 standalone evaluate MUST 对 beam Top-K、ADBA/DBA 和公开 top-level scalar 使用同一套 selected metric horizons。配置或 runtime 解析出的 `metric_horizons` MUST 被记录在 metrics metadata 中，subset top-level scalar MUST NOT 回退到 first valid slot 口径。

#### Scenario: subset top1 使用 selected horizons
- **WHEN** 配置选择 `metric_horizons=[2,4,6]` 或等价 horizon 集合
- **THEN** 普通 validation 的 top-level Top-1 MUST 基于这些 selected horizons 聚合
- **AND** force-mask subset validation 的 top-level `top1` 或等价 scalar MUST 使用同一 selected horizon 聚合
- **AND** subset validation MUST NOT 使用 first valid slot 作为 top-level `top1`

#### Scenario: standalone evaluate 记录同一口径
- **WHEN** 用户通过 standalone evaluate 运行同一配置
- **THEN** evaluate metrics/report MUST 记录实际使用的 `metric_horizons`
- **AND** Top-K 与 DBA/ADBA top-level scalar MUST 与训练验证使用同一 horizon 选择规则
- **AND** 若输出逐 horizon 诊断，诊断字段 MUST 与 top-level 聚合字段可区分

#### Scenario: 未配置 horizons 使用统一默认
- **WHEN** 配置没有显式设置 `metric_horizons`
- **THEN** validation、subset validation 和 evaluate MUST 使用同一个默认 horizon 集合
- **AND** metrics metadata MUST 记录默认来源或等价说明

### Requirement: 主结论过滤 split eligibility
实验 summary、quick conclusion 和横向比较工具 MUST 消费 split eligibility metadata。任何使用 unknown 或 leakage diagnostics 失败的 split 的 run MUST 不被用于 strict validation 主结论，除非用户显式请求 debug/sanity 汇总。

#### Scenario: strict split run 可进入主结论
- **WHEN** run metadata 记录 `strict_validation_eligible=true`
- **THEN** summary MAY 将该 run 纳入 strict validation 横向比较
- **AND** summary MUST 保留 split strategy、split metadata 路径和样本数，便于复核可比性

#### Scenario: strict-ineligible split run 被排除
- **WHEN** run metadata 记录 `strict_validation_eligible=false`
- **THEN** summary MUST 将该 run 排除出 strict 主结论
- **AND** summary MUST 记录 exclusion reason 和 split metadata 路径
- **AND** 用户仍 MAY 在 debug/sanity 视图中查看该 run 的原始指标

#### Scenario: split metadata 缺失时保守处理
- **WHEN** summary 读取到没有 split metadata 的 MMW Town10 run
- **THEN** summary MUST 标记该 run 的 split eligibility 为 unknown
- **AND** strict 主结论 MUST 默认排除该 run
- **AND** 输出 MUST 给出生成或引用 strict split metadata 的修复提示

### Requirement: 默认实验入口去 KD-first 化
项目默认 quickstart、README 推荐入口、当前主线 quick validation 和新 canonical mainline 配置 MUST 以 supervised/adaptation、JEPA、CSI hardening、baseline/control 或当前诊断工作流为默认。旧 KD、BGAM 和 viewer manifest 配置不得作为当前主线默认实验入口。

#### Scenario: README quickstart 使用当前主线
- **WHEN** 开发者阅读 README 或当前主线运行说明
- **THEN** 推荐的首个训练、评估或诊断命令 MUST 使用当前 supervised/adaptation、JEPA、CSI、baseline/control 或当前诊断配置
- **AND** 文档 MUST 不把 `logits_kd`、`rkd`、Hist/HiST、standalone Top8 selector、GPS residual 或 camera residual 作为当前主线 quickstart

#### Scenario: canonical mainline 配置不要求 teacher checkpoint
- **WHEN** 用户加载当前推荐的 mainline 配置
- **THEN** 配置 MUST 能在没有 teacher checkpoint 的情况下完成解析和 dry-run/smoke 构建
- **AND** 输出 metadata MUST 不记录 KD-enabled lineage

### Requirement: 项目描述反映当前主线
项目元数据、README 和高层文档 MUST 将当前项目主线描述为多模态 beam prediction、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、GPS v2/adapter、MMW Town GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理和诊断，而不是 KD-first、HiST-Beam-first、Raymobtime-first、Top8/residual-first、BGAM-first、viewer-first 或 GPS coarse-anchor-first 工作流。历史 KD、Hist、Raymobtime、Top8 selector、residual、camera residual、BGAM、viewer manifest 或 GPS coarse anchor 背景可以保留在 archive 或历史说明中，但必须标记为已退役或历史记录。

#### Scenario: pyproject 描述不再 KD Hist 或退役路线 first
- **WHEN** 开发者查看 `pyproject.toml` 的项目 description
- **THEN** description MUST 不把 knowledge distillation、HiST-Beam、Top8 selector、residual 或 GPS coarse anchor 描述为当前唯一或首要工作流
- **AND** 若提到这些路线，MUST 表达其为 legacy、historical 或 retired

#### Scenario: 文档保留历史说明
- **WHEN** README 或 docs 提到历史 KD、Hist、Top8 selector、residual、camera residual 或 GPS coarse anchor 代码
- **THEN** 文档 MUST 说明对应能力已从当前 active mainline 退役
- **AND** 文档 MUST 不提供当前推荐运行命令

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理和当前诊断。KD baseline、HiST-Beam/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS coarse anchor、residual fusion、camera residual、BGAM、viewer manifest、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、supporting、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、Raymobtime s008、retired Top8 selector/residual/GPS coarse anchor 命令或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、Top8 selector、residual、camera residual、GPS coarse anchor、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/JEPA/CSI 主线

#### Scenario: 当前 workflow 文档声明运行状态
- **WHEN** 文档列出当前实验配置、benchmark manifest 或诊断配置
- **THEN** 文档 MUST 标明该条目是 formal、lowmem、smoke、debug、evaluation-only、upper-bound、historical ablation 还是 mock
- **AND** upper-bound、mock、smoke 或 historical ablation MUST 不得被写成正式结论

### Requirement: 健康检查反映保留入口
快速健康检查 MUST 覆盖当前仍支持的架构边界、包内 CLI、JEPA visual analysis、GPS shortcut benchmark、文档健康和当前主线 focused tests。健康检查 MUST 不要求 Raymobtime s008、已退役的模态失衡诊断脚本、fusion KD virtual alias、BGAM、viewer manifest 或 HiST-Beam/Hist CLI 可用。

#### Scenario: focused validation 不依赖退役入口
- **WHEN** 开发者执行本 change 的 focused 验证
- **THEN** 验证命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 不包含已退役的 Hist CLI、Hist configs 或独立模态诊断脚本
- **AND** 验证 MUST 覆盖配置加载失败、架构边界、registry 和保留 evaluation subset 能力

### Requirement: MMW Town GPS v2 CLI workflow
项目 MUST 提供配置驱动的 MMW Town GPS-only v2 runner、plotter 和 comparison 入口。入口 MUST 位于 `kd_sensing` 包内并可通过 console script 或 `python -m kd_sensing.cli.<module>` 运行；项目 MUST NOT 要求用户通过 `python -m src.*` 调用该 workflow。

#### Scenario: v2 runner help 可用
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--label-space`、`--target-scene`、`--support-ratio`、`--support-num` 和 `--support-mode`

#### Scenario: plotter 和 comparison help 可用
- **WHEN** 用户执行 v2 plotter 或 comparison console script 的 `--help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 results dir、previous dir 或 new dir 等必要参数

### Requirement: MMW Town GPS v2 default configuration
项目 MUST 提供 `configs/mmw_town_gps_adapter_v2.yaml` 或等价 v2 配置。配置 MUST 声明数据根、已有分析目录、label space、四个 scene、num_beams、split、model、loss、train、adapt、metrics 和 ablation 矩阵。

#### Scenario: 默认配置可解析
- **WHEN** 用户通过 v2 runner 传入默认 v2 配置
- **THEN** 系统 MUST 能解析完整配置
- **AND** 默认 label space MUST 为 `mapping_enabled`
- **AND** 默认 scene 列表 MUST 覆盖 crossroad、skybridge、curvyroad 和 Hroad

### Requirement: README documents MMW Town GPS v2
README MUST 增加 MMW Town GPS-only v2 说明，覆盖普通跨场景 GPS 分类器失败原因、circular beam distance、mapping_enabled/mapping_disabled、SceneAdapterV2 三种 adapter、完整实验命令、summary_by_scene 解读、crossroad/Hroad 残差诊断和后续多模态 residual correction 边界。

#### Scenario: README 提供可执行命令
- **WHEN** 开发者阅读 README 的 MMW Town GPS-only v2 小节
- **THEN** 文档 MUST 提供使用 `conda run -n kd_mm_beam` 的 runner、plotter 和 comparison 命令
- **AND** 文档 MUST 明确本 change 不实现多模态 residual correction

### Requirement: GPS+LiDAR BGAM workflow 已从实验入口退役
当前训练、评估、quickstart、CLI help、run metadata 和推荐文档 MUST 不再包含 GPS+LiDAR BGAM workflow。旧 BGAM 配置路径、console script、manifest enrich、dataset、model、loss、engine、debug mask 和 focused tests 不得作为当前 workflow 兼容承诺。

#### Scenario: BGAM 配置和命令不存在
- **WHEN** 开发者检查配置、pyproject entry points 和包内 CLI
- **THEN** 项目 MUST 不保留 `configs/deepsense6g_gps_lidar_bgam.yaml` 或 `configs/mmw_town_gps_lidar_bgam.yaml`
- **AND** 项目 MUST 不声明 GPS+LiDAR BGAM prepare/run/evaluate 相关 `kd-sensing-*` 命令
- **AND** 项目 MUST 不保留等价 virtual config 或 thin alias

#### Scenario: BGAM 实现和测试不存在
- **WHEN** 开发者检查 `src/kd_sensing` 和 `tests`
- **THEN** 项目 MUST 不保留 `gps_lidar_bgam` 专属 data、engine、model、loss 或 diagnostics 模块
- **AND** 项目 MUST 不保留 `tests/test_gps_lidar_bgam_*.py` focused tests
- **AND** 最终回归仍 MUST 使用 `conda run -n kd_mm_beam pytest -q`

#### Scenario: README 说明退役而非运行
- **WHEN** README 或实验矩阵提到 GPS+LiDAR BGAM
- **THEN** 文档 MUST 明确其已退役或仅作为历史背景
- **AND** 文档 MUST 指向当前 MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark 或其它 current workflow

### Requirement: Hist workflow 已从当前实验入口退役
当前训练、评估、quickstart、CLI help、run metadata 和推荐文档 MUST 不再包含 HiST-Beam/Hist LOSO 入口。旧 Hist 配置路径、console script 和 run plan 不得作为当前 workflow 兼容承诺。

#### Scenario: CLI help 不包含 Hist 保留入口
- **WHEN** 开发者执行当前推荐的 CLI help 验证
- **THEN** `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`、`kd-sensing-runs`、`kd-sensing-jepa-visual-analysis` 和 `kd-sensing-jepa-gps-shortcut-benchmark` MUST 正常退出
- **AND** 验证 MUST 不要求 `kd-sensing-hist-beam-loso` 存在

#### Scenario: 旧 Hist 配置路径失败
- **WHEN** 用户传入 `configs/hist_beam/quick_smoke.yaml` 或其它 `configs/hist_beam/` 路径
- **THEN** 配置加载 MUST 失败或报告路径已退役
- **AND** 系统 MUST 不生成等价 virtual config

### Requirement: 配置驱动 JEPA 预训练
项目 MUST 支持通过现有训练 CLI 运行 GPS-conditioned JEPA 预训练配置。该配置 MUST 使用 `model.primary` 构建 JEPA 主模型，MUST 通过 image/GPS 模态契约准备输入，MUST 使用 `gps_conditioned_jepa` objective 计算自监督 latent loss，并 MUST 不构建 distiller、旧 teacher/student KD runtime 或外部 frozen teacher checkpoint。

#### Scenario: 使用配置启动 JEPA 预训练
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-train --config <jepa_config>`
- **THEN** 系统 MUST 构建包含 image 和 GPS 输入的 dataset/dataloader
- **AND** 系统 MUST 构建 `model.primary` 指定的 GPS-conditioned JEPA 模型
- **AND** 系统 MUST 以 JEPA latent loss 完成 forward、backward、optimizer step 和 EMA target encoder update

#### Scenario: JEPA 配置不启用 KD
- **WHEN** 用户加载 JEPA 预训练配置
- **THEN** 配置 MUST 不包含 `distillation.*`、`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd` 或旧 KD alias
- **AND** 配置加载和训练构建流程 MUST 不构建 distiller 或 frozen teacher checkpoint

### Requirement: JEPA 验证与运行产物
JEPA 预训练 workflow MUST 在验证阶段计算 `val_jepa_loss`，并 MUST 将运行产物写入当前统一输出目录。运行目录 MUST 保存解析后的配置、训练日志、checkpoint、TensorBoard 标量、运行状态和 objective metadata。固定 `output.run_name` 的唯一目录、resume 和 overwrite 语义 MUST 与现有训练 workflow 保持一致。

#### Scenario: JEPA validation 输出
- **WHEN** JEPA 训练完成一个 epoch 并进入验证
- **THEN** validation MUST 计算并返回 `val_jepa_loss`
- **AND** validation MUST 不要求 `target_beam`
- **AND** validation MUST 不写出 beam Top-K、DBA 或 beam prediction report

#### Scenario: JEPA final config metadata
- **WHEN** JEPA 训练写出最终运行配置
- **THEN** `final_config.yaml` MUST 记录解析后的 `experiment.objective: gps_conditioned_jepa`
- **AND** runtime metadata MUST 记录 JEPA loss、主 metric、metric mode、mask sampler、EMA decay 和 context encoder artifact 信息

#### Scenario: JEPA resume 复用运行目录
- **WHEN** 用户设置 `training.resume: true` 并提供 JEPA `output.run_name`
- **THEN** 训练流程 MUST 从该运行目录恢复 checkpoint
- **AND** 恢复 MUST 包含 context encoder、target encoder、optimizer、scheduler、epoch 和 best `val_jepa_loss`

### Requirement: JEPA canonical smoke 配置
项目 MUST 提供一个用于快速验证的 GPS-conditioned JEPA 配置。该配置 MUST 使用当前保留的数据集和模态契约，默认启用 image RGB/ImageNet profile 和 GPS relative-polar feature，默认输出到 `outputs/`，并 MUST 能通过小 epoch/小 batch override 运行 smoke test。

#### Scenario: JEPA smoke 配置可加载
- **WHEN** 开发者加载 JEPA canonical smoke 配置
- **THEN** 配置 MUST 解析为 `experiment.objective: gps_conditioned_jepa`
- **AND** `model.primary.type` MUST 为已注册 JEPA 模型
- **AND** image 与 GPS 输入 profile MUST 与模态契约一致

#### Scenario: JEPA smoke test
- **WHEN** 开发者使用 synthetic 或小比例数据运行 1 epoch JEPA smoke
- **THEN** 训练 MUST 完成 forward、loss、backward、optimizer step、EMA update、validation 和 checkpoint 保存
- **AND** 运行产物 MUST 包含 `val_jepa_loss`

### Requirement: JEPA paper-split full 配置
项目 MUST 提供主 JEPA 预训练和 GPS-biased mask ablation 的 low-memory full 配置。该配置 MUST 使用 DeepSense6G scenes 32、33、34 作为训练来源，并 MUST 将 scenes 31、32、33、34 纳入验证/监控集合。多场景训练运行产物 MUST 记录 source scene 列表与每个 scene 的样本数，避免被误判为 scene31-only 实验。

#### Scenario: JEPA full 配置使用多场景训练
- **WHEN** 用户加载 JEPA full low-memory 配置
- **THEN** `data.dataset.train_scenes` MUST 包含 32、33 和 34
- **AND** `data.dataset.test_scenes` MUST 包含 31、32、33 和 34
- **AND** 输出 run name MUST 明确区分该 run 是 scenes32-34 训练口径

### Requirement: 现有 supervised/adaptation workflow 不变
新增 JEPA 预训练 workflow MUST 不改变现有 beam、occlusion、position、multitask、GPS v2、CSI hardening 或 supervised fusion workflow 的默认配置和指标。Raymobtime s008、legacy KD、standalone Top8 selector、residual、BGAM 和 viewer 路线仍只作为退役或 supporting guard 语义保留，不属于当前默认 workflow。

#### Scenario: 默认 beam 配置行为不变
- **WHEN** 用户加载未设置 `experiment.objective` 的现有 supervised beam 配置
- **THEN** 系统 MUST 继续默认使用 `beam` objective
- **AND** 系统 MUST 继续计算 beam loss、Top-K、DBA 和 `val_adba`

#### Scenario: 旧 KD 入口仍被拒绝
- **WHEN** 用户请求旧 `logits_kd`、`rkd`、`teacher_no_kd` 或 retired fusion KD 配置
- **THEN** 系统 MUST 继续拒绝该配置
- **AND** 错误信息 MUST 继续指向当前 supervised/adaptation 或 JEPA 预训练入口，而不是恢复旧 KD workflow

### Requirement: BeamBench 对齐 supervised 下游验证
项目 MUST 提供 image+GPS supervised fair low-memory 配置族，用于比较 supervised baseline 与 JEPA context encoder 初始化的下游 beam prediction。该配置族 MUST 使用 DeepSense6G scenes 32、33、34 的训练 split 作为训练来源，MUST 使用训练 split 内部划分的 validation 子集做 checkpoint selection，MUST 在训练完成后单独评估 scenes 31、32、33、34 的 test split，并 MUST 将 final test metrics 写入运行 metadata。保留 `beambench_fair` 文件名的配置 MUST 表示 BeamBench Table III 的输入、split、target 和 metric 口径对齐，不得继续使用旧的 8 帧 relative-polar fair 口径。

#### Scenario: fair 配置使用独立选模 split
- **WHEN** 用户加载 BeamBench-fair supervised 下游配置
- **THEN** `data.validation_from_train.enabled` MUST 为 true
- **AND** 训练循环 MUST 优先使用 `validation` dataloader 计算 early stopping metric
- **AND** `test` dataloader MUST 不用于 early stopping/checkpoint selection

#### Scenario: fair 配置训练后执行 final test
- **WHEN** fair supervised 训练结束且运行目录存在 `checkpoints/best.pth`
- **THEN** 系统 MUST 重新加载该 checkpoint
- **AND** 系统 MUST 在 `test` dataloader 上计算 final test metrics
- **AND** runtime metadata MUST 记录 `final_test_metrics.evaluation_split: test`
- **AND** runtime metadata MUST 记录 `final_test_metrics.model_selection_split`

#### Scenario: fair 配置使用 BeamBench DBA 口径
- **WHEN** fair supervised 配置计算 DBA 或 ADBA
- **THEN** `evaluation.dba_distance_mode` MUST 支持并设置为 `linear`
- **AND** linear 模式 MUST 使用非环形 beam index 距离
- **AND** 未显式设置该字段的现有配置 MUST 继续使用 circular DBA 默认行为

#### Scenario: fair 配置固定 BeamBench 输入和预测窗口
- **WHEN** fair supervised 配置被加载
- **THEN** `data.dataset.num_pred` 和 `model.num_pred` MUST 为 1
- **AND** `data.dataset.seq_len` 和 `model.seq_length` MUST 为 1
- **AND** GPS 输入 MUST 使用 `paper_distance_angle` 二维 Direct 特征
- **AND** beam target MUST 设置为 `beam_target_source: current`
- **AND** `model.primary.gps_input_size` MUST 为 2
- **AND** scene paper calibration angle MUST 通过 `gps_angle_offset_source: paper_scene_default` 或等价运行 metadata 记录
- **AND** `evaluation.k_values` MUST 为 `[1, 3, 5]`
- **AND** scheduler MUST 设置为 `none`
- **AND** Table III Camera AE+GPS Direct 数值复现 MUST 使用专用 BeamBench runner，不得把通用 Image+GPS/JEPA fair 配置伪装成 Table III row 模型

### Requirement: 2604.05668 对齐 supervised 下游验证
项目 MUST 提供 image+GPS supervised 2604 对齐配置族，用于与 arXiv:2604.05668 的 S32/S33/S34 主表口径比较。该配置族 MUST 合并 DeepSense6G scenes 32、33、34 的官方 train/test labeled CSV，MUST 在每个 scene 内按 beam label 做固定 seed 的 80/10/10 stratified train/validation/test split，MUST 使用 `seq_len: 5` 和 `num_pred: 1`，并 MUST 记录 split protocol 与每个 split 的样本数。

#### Scenario: 2604 配置使用合并后 stratified split
- **WHEN** 用户加载 2604 对齐 supervised 配置
- **THEN** `data.dataset.split_protocol` MUST 为 `stratified_80_10_10`
- **AND** `data.dataset.train_scenes`、`validation_scenes` 和 `test_scenes` MUST 包含 32、33 和 34
- **AND** train/validation/test MUST 来源于每个 scene 的 `train_seqs_RA_GPS_LIDAR.csv` 与 `test_seqs_RA_GPS_LIDAR.csv` 合并集合

#### Scenario: 2604 配置匹配历史窗口
- **WHEN** 2604 对齐 supervised 配置被加载
- **THEN** `data.dataset.seq_len` 和 `model.seq_length` MUST 为 5
- **AND** `data.dataset.num_pred` 和 `model.num_pred` MUST 为 1

#### Scenario: 2604 split 使用 train-only normalization
- **WHEN** 2604 split protocol 构建 image+GPS dataloader
- **THEN** GPS scaler MUST 只从 stratified train 子集拟合
- **AND** validation/test MUST 复用该 scaler
- **AND** runtime metadata MUST 记录 scaler 来源与 split protocol

### Requirement: 当前推荐 workflow 排除 Top8 residual coarse 路线
README、实验矩阵、quickstart、docs inventory 和健康检查 MUST 不再把 Top8 selector、standalone Top8 candidate manifest、GPS coarse anchor、GPS prior residual correction、camera residual、BGAM 或 viewer manifest 描述为当前可运行或推荐 workflow。当前推荐面 MUST 聚焦仍保留的 supervised/adaptation、Image+GPS JEPA、GPS v2/adapter、MMW GPS v2、CSI hardening、Vision-Position baseline、Arnold22 Camera AE+GPS Direct、预处理和当前诊断。

#### Scenario: quickstart 不展示退役命令
- **WHEN** 开发者阅读 README、README_REPRODUCE 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不提供退役 Top8 selector/residual/GPS coarse anchor 命令作为当前运行步骤
- **AND** 文档 MUST 指向仍存在的配置化 CLI 和保留 workflow

#### Scenario: 健康检查不要求退役入口
- **WHEN** 开发者执行快速健康检查或架构边界测试
- **THEN** 检查 MUST 不要求退役 console scripts、配置、CLI、engine、model 或 loss 可导入
- **AND** 检查 MAY 断言这些入口已不存在

### Requirement: 推荐实验工作流不包含 Raymobtime s008
README、实验矩阵、快速健康检查和配置驱动 workflow 文档 MUST 不再把 Raymobtime s008 作为当前推荐或可运行实验。历史 archive MAY 保留 Raymobtime 记录，但 MUST 不作为当前入口、教程或验证命令。

#### Scenario: README 和实验矩阵移除 Raymobtime 入口
- **WHEN** 用户阅读 README、docs/experiment_matrix.md 或研究笔记中的当前推荐流程
- **THEN** 文档 MUST 不再推荐运行 Raymobtime s008 预处理、训练或评估命令
- **AND** 文档 MUST 明确当前主线使用仍保留的数据集和 viewer workflow

#### Scenario: 健康检查不要求 Raymobtime focused test
- **WHEN** 开发者执行快速验证说明中的 focused tests
- **THEN** 验证命令 MUST 不要求 `tests/test_raymobtime_s008_selection.py`
- **AND** 验证 MUST 覆盖通用 CLI、架构边界、配置退役 guard 和当前保留 workflow

#### Scenario: 旧 Raymobtime 配置不可作为 workflow
- **WHEN** 用户传入 `configs/raymobtime/` 或 `configs/preprocess/raymobtime_s008_*.yaml` 下的旧配置路径
- **THEN** 系统 MUST 拒绝该 workflow 或这些配置文件 MUST 已被删除
- **AND** 错误信息 MUST 指出 Raymobtime s008 已退役

### Requirement: 多场景训练输出 scope
训练和默认评估流程 MUST 能为 DeepSense6G 多场景协议生成稳定 scenegroup scope。配置包含 `train_scenes`、`validation_scenes`、`test_scenes` 或 `eval_scenes`，且有效 scene 集合不是单个 scene 时，默认输出根 MUST 使用 `outputs/scenegroup_<scene-range-or-list>/` 或用户显式配置的等价根目录。

#### Scenario: S32-S34 多场景训练输出
- **WHEN** 配置声明 `train_scenes: [32, 33, 34]` 且 `output.dir: outputs`
- **THEN** 默认训练运行目录 MUST 创建在 `outputs/scenegroup_s32_s34/<run_name>/`
- **AND** final config runtime metadata MUST 记录 scene scope、source scenes、validation scenes 和 test scenes

#### Scenario: S31-S34 多场景评估输出
- **WHEN** 配置声明评估覆盖 scenes 31、32、33、34 且未显式传入完整 `--output-dir`
- **THEN** 默认评估集合 MUST 写入 `outputs/evaluations/<study_id>/` 或 `outputs/scenegroup_s31_s34/evaluation_<run_name>_<timestamp>/`
- **AND** 输出 metadata MUST 能区分训练 source scenes 与 evaluation scenes

#### Scenario: 显式输出目录仍保持完整路径
- **WHEN** 用户通过训练配置 `output.dir` 或评估入口 `--output-dir` 显式传入完整输出目录
- **THEN** 系统 MUST 尊重该路径
- **AND** 系统 MUST 不额外追加 scene 或 scenegroup 片段

### Requirement: 优先退役 workflow 不得作为当前实验入口
当前实验 workflow MUST 不再推荐、声明或验证 AMR-Net_gps_image mock/source-audit runner、JEPA-MSAC mock/paper-aligned runner、MMW GPS v2 旁支 `scripts/mmw/visualize_gps_*` 脚本，或非 CSI 的本地 shell orchestration 脚本。历史背景 MAY 保留，但 MUST 不提供 current 运行命令。

#### Scenario: 实验矩阵不推荐退役 workflow
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-run-amr-net-gps-image`
- **AND** 文档 MUST 不推荐运行 `kd-sensing-run-jepa-msac`
- **AND** 文档 MUST 不推荐运行被退役的 MMW 旁支诊断脚本或非 CSI shell orchestration 脚本

#### Scenario: 配置加载拒绝退役配置
- **WHEN** 用户加载 `configs/baselines/amr_net_gps_image.yaml`、`configs/pretraining/jepa_msac_s32_smoke.yaml` 或 `configs/pretraining/jepa_msac_s32_paper.yaml`
- **THEN** 配置加载 MUST 失败或对应实体配置 MUST 不存在
- **AND** 错误信息或文档 MUST 说明该 workflow 已退役并指向当前 baseline、diagnostic 或 reproduction 入口

### Requirement: 当前替代入口必须清晰
退役上述 workflow 后，项目 MUST 在文档中给出当前替代入口。替代入口 MUST 是仍受支持的 package CLI、current config 或明确保留的 shell runner，不得新增旧式兼容 wrapper。

#### Scenario: MMW 诊断迁移到 package CLI
- **WHEN** 文档说明 MMW GPS v2 图表或对比
- **THEN** 文档 MUST 指向 `kd-sensing-plot-mmw-town-gps-v2` 和 `kd-sensing-compare-mmw-town-gps-v2`
- **AND** 文档 MUST 不要求用户直接运行退役的 `scripts/mmw/visualize_gps_*` 脚本

#### Scenario: shell runner 迁移到当前入口
- **WHEN** 文档说明 DeepSense GPS soft-label、MMW soft-label ablation 或 MMW sunny modal15 历史实验
- **THEN** 文档 MUST 将其标记为 historical 或 retired
- **AND** 当前运行建议 MUST 使用 `kd-sensing-train`、当前 package diagnostics、保留的 CSI hardening matrix runner 或明确 current 的配置

### Requirement: Physics-informed MMW workflow entrypoints
系统 MUST 通过现有 `kd-sensing-train`、`kd-sensing-evaluate` 和包内 CLI 支持 physics-informed MMW baseline。项目 MUST 不新增仓库根训练/评估脚本或 `scripts/*.py` thin alias；dataset inspection MUST 作为包内 CLI、console script 或训练 debug shape summary 实现。

#### Scenario: 通过训练 CLI 启动 physics-informed debug 配置
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/physics_informed_mmw_debug.yaml`
- **THEN** 系统 MUST 构建 `mmw` dataset、`pinn_multimodal_beam` primary model、physics-informed loss 和现有 optimizer/checkpoint/runtime
- **AND** 系统 MUST 不调用根目录 `train.py`

#### Scenario: inspection 不使用 scripts thin alias
- **WHEN** 用户需要检查 MMW physics 字段和 shape
- **THEN** 系统 MUST 提供包内 CLI 或 debug shape summary
- **AND** 文档 MUST 不推荐 `python scripts/inspect_dataset.py`

### Requirement: Physics-informed ablation configs
系统 MUST 提供可配置 ablation workflow，用于关闭 physics loss、CSI reconstruction、path loss、array consistency 或 physics head，并提供 `vision_only`、`partial_csi_multimodal`、`history_csi_multimodal` 和 `oracle_full_csi` 四类 CSI 使用设置。ablation 配置 MUST 复用同一 dataset split、beam label space、output boundary 和 training CLI。

#### Scenario: no physics ablation
- **WHEN** 用户加载 no-physics ablation 配置
- **THEN** final config MUST 将 CSI/path/array/alignment physics loss 权重置零
- **AND** model metadata MUST 记录 physics branch 或 physics head 被关闭

#### Scenario: modality ablation
- **WHEN** 用户加载 CSI-only、image-only、image+CSI 或 full multimodal 配置
- **THEN** dataset 和 model MUST 只要求配置声明的启用模态
- **AND** 未启用模态的缺失文件 MUST 不阻止实验启动

#### Scenario: leakage-safe CSI 实验配置
- **WHEN** 用户加载 `physics_informed_mmw_vision_only.yaml`
- **THEN** 配置 MUST 设置 `use_csi_input=false` 且模型不启用 CSI 输入
- **WHEN** 用户加载 `physics_informed_mmw_partial_csi_multimodal.yaml` 或 `physics_informed_mmw_history_csi_multimodal.yaml`
- **THEN** 配置 MUST 启用多模态 sensing 加受限 CSI 输入，并将当前完整 CSI 仅用于监督
- **WHEN** 用户加载 `physics_informed_mmw_oracle_full_csi.yaml`
- **THEN** 配置 MUST 设置 `csi_input_mode=oracle_full` 和 `allow_oracle_full_csi_input=true`

### Requirement: Physics workflow artifacts and documentation
系统 MUST 将 physics-informed run 的 final config、metrics、loss breakdown、shape summary、sensitive usage flags 和 claim status 写入现有运行产物或文档索引。README MUST 只提供简短入口和数据/产物边界；详细实验口径 MUST 进入现有 docs 主线实验文档。

#### Scenario: 运行产物记录物理字段
- **WHEN** physics-informed 训练或评估完成
- **THEN** final config 或 run metadata MUST 记录 enabled modalities、physics losses、array/codebook config、shape summary 和 main-conclusion eligibility
- **AND** metrics MUST 包含普通 beam 指标和可用的 physics metrics

#### Scenario: 文档不声明未验证 claim
- **WHEN** 文档新增 physics-informed MMW baseline 条目
- **THEN** result claims registry MUST 将真实性能 claim 标记为 pending 或 unverified，直到有可追溯运行产物
- **AND** 文档 MUST 不把 synthetic smoke 或 debug run 写成正式实验结果

### Requirement: Sparse-pilot physics-informed MMW config
系统 MUST 提供 sparse-pilot physics-informed MMW 配置，使用现有 `kd-sensing-train` 入口运行，不新增根脚本。该配置 MUST 启用 image + sparse pilot CSI 输入，并保持完整 CSI/path/beam power 作为训练监督或诊断。

#### Scenario: sparse pilot 配置加载
- **WHEN** 用户加载 `configs/fusion/physics_informed_mmw_sparse_pilot_multimodal.yaml`
- **THEN** 配置 MUST 设置 `data.csi_input_mode=sparse_pilot`
- **AND** `model.primary.modalities` MUST 包含 `image` 和 `csi`
- **AND** `oracle_full` 仍 MUST 只作为 upper-bound 配置

### Requirement: Paper-style physics MMW experiment configs
项目 SHALL provide opt-in configs for the paper-style physics-informed MMW baseline. These configs MUST use current `kd-sensing-train` / evaluation entry points, MUST keep sparse or restricted wireless observation separate from full CSI supervision, and MUST record whether the run is formal, debug/smoke, or oracle upper-bound.

#### Scenario: sparse-pilot multimodal 配置
- **WHEN** 用户加载 paper-style sparse-pilot multimodal physics MMW 配置
- **THEN** final config MUST enable `model.primary.type=pinn_multimodal_beam`
- **AND** model config MUST enable paper-style tokenizer frontend and shared Transformer fusion
- **AND** data config MUST use restricted CSI input such as `sparse_pilot` rather than default full current CSI

#### Scenario: debug 配置不可进入正式结论
- **WHEN** debug/smoke 配置允许随机初始化 `jepa_context_image` 或使用 synthetic batch
- **THEN** run metadata MUST mark `formal_experiment_eligible=false`
- **AND** report MUST NOT be treated as formal paper-style baseline evidence

#### Scenario: oracle 配置明确标记
- **WHEN** oracle full CSI input is explicitly enabled for upper-bound comparison
- **THEN** run metadata MUST mark `oracle_upper_bound=true`
- **AND** run metadata MUST mark `main_conclusion_eligible=false`
- **AND** summary MUST state that current full CSI was used as model input

### Requirement: Paper-style physics MMW validation
Paper-style physics MMW implementation MUST include focused validation that does not depend on real `dataset/` contents. Validation MUST cover config loading, registry build, synthetic forward, physics loss/backward, output adaptation, metadata, and shape handling for `[B, T, Nsc, Nant, 2]` CSI targets.

#### Scenario: synthetic forward/loss smoke
- **WHEN** focused tests construct a synthetic paper-style physics MMW batch
- **THEN** model forward MUST produce finite logits, `path_hat` and `h_hat`
- **AND** physics-informed loss MUST complete backward with finite gradients
- **AND** tests MUST NOT read real `dataset/`

#### Scenario: JEPA image tokenizer without GPS context smoke
- **WHEN** focused tests build the image tokenizer path
- **THEN** test config MUST use `jepa_context_image` with a non-GPS pooler
- **AND** forward MUST succeed without `gps_condition_features`
