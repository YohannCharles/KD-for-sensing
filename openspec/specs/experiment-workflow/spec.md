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

### Requirement: BTAPA tau1 seed 与 es20 配置族
项目 MUST 提供不覆盖原始 tau1 的 BTAPA tau1 seed2/seed3 配置和 es20 配置族。除 seed、输出路径和 es20 early stopping 字段外，配置 MUST 与 `main_v3_strong_reliability_btapa_tau1.yaml` 保持一致，并 MUST 不启用 RBMA、JEPA、KD、fullaux 或 ADBA-aware proto。

#### Scenario: seed 配置不覆盖原始 run
- **WHEN** 用户运行 `main_v3_strong_reliability_btapa_tau1_seed2.yaml` 或 `main_v3_strong_reliability_btapa_tau1_seed3.yaml`
- **THEN** 输出路径或 run name MUST 包含 `btapa_tau1_seed2` 或 `btapa_tau1_seed3`
- **AND** 配置 MUST 支持 `--auto_resume`

#### Scenario: es20 配置启用短训练早停
- **WHEN** 用户运行任一 `main_v3_strong_reliability_btapa_tau1_es20*.yaml`
- **THEN** 配置 MUST 设置 `max_epochs: 20` 或项目等价字段
- **AND** 配置 MUST 启用以 `val_top1` 或项目等价字段为指标的 early stopping、patience 5 和 best checkpoint 选择

### Requirement: 固定 GPU shell launcher 已收敛为直接命令
项目 MUST 不再要求固定 GPU queue shell 来运行 BTAPA tau1、proto-vs-BTAPA 或 night-grid 训练。保留的契约是配置、manifest、`kd-sensing-train --config <yaml>`、fresh eval helper 和只读分析脚本；并发、GPU 绑定和日志策略属于用户本地任务系统或 shell 临时命令，不进入源码长期表面。

#### Scenario: BTAPA tau1 直接训练
- **WHEN** 用户需要复跑 BTAPA tau1 seed/es20 配置
- **THEN** 用户 MUST 使用 `kd-sensing-train --config configs/scene31/<btapa-yaml> --auto_resume` 或等价当前训练入口
- **AND** 项目 MUST 不要求 `scripts/run_btapa_tau1_validation.sh` 或 `scripts/run_proto_vs_btapa_8gpu.sh` 存在

#### Scenario: fresh eval 仍可组合
- **WHEN** 用户需要训练后 apples-to-apples 复评
- **THEN** 用户 MAY 运行 `scripts/reevaluate_apples_to_apples.py`、`scripts/eval_night_grid.py` 或对应 analysis helper
- **AND** 缺失 checkpoint 的 run MUST warning 但不阻断其它 run 复评

### Requirement: proto vs BTAPA seed mean±std 分析
项目 MUST 提供 `scripts/analyze_proto_vs_btapa_seeds.py`，读取 fresh apples-to-apples eval 输出，生成 seed metrics、mean±std、delta 和 paper-ready observation。报告 MUST 重点列出 full、avg_missing、missing_gps、radar_only、lidar_only 的 Top-1 和 avg_missing ADBA，并在 delta 小于 std 时提示谨慎报告。

#### Scenario: 输出 mean std 与 delta
- **WHEN** 用户传入 proto 三 seed、BTAPA tau1 三 seed 和 fresh eval 目录
- **THEN** 脚本 MUST 输出 seed metrics、mean±std、delta mean 和 Markdown 报告
- **AND** Markdown MUST 包含保守 paper-ready observation 与 seed 方差提示

### Requirement: Scene31 night grid config generation
项目 MUST 提供 `scripts/generate_experiment_grid.py`，从 `configs/scene31/templates/main_v3_proto_es20_base.yaml` 生成 A-F 共 58 个 run 配置，并在 manifest 中加入 6 个 proto/BTAPA reference run，总计 64 个 run。默认 MUST 不覆盖已有配置。

#### Scenario: 生成 manifest
- **WHEN** 用户运行生成脚本并指定 out_dir
- **THEN** 系统 MUST 写出 `experiment_manifest.csv` 和 `experiment_manifest.json`
- **AND** manifest MUST 包含 `run_name,group,config_path,seed,method_tags,expected_epochs,priority`

#### Scenario: 输出路径唯一
- **WHEN** 生成任一 night grid 配置
- **THEN** 配置中的 run name、exp name 或 output_dir MUST 与其它 run 唯一区分

### Requirement: night grid generated configs are local artifacts
项目 MUST 保留 night-grid manifest/base/generator 和 generator sanity test，但 MUST 不要求把生成的 58 个 run YAML 长期提交到源码。需要训练时，用户先在本地输出目录或显式 config 目录生成 YAML，再使用当前 `kd-sensing-train` 入口运行。

#### Scenario: 生成后训练
- **WHEN** 用户运行 `scripts/generate_experiment_grid.py --out_dir <local-config-dir>`
- **THEN** generator MUST 写出 manifest 和实体 YAML 到指定目录
- **AND** 源码长期表面 MAY 只保留 manifest/base/generator

### Requirement: night grid fresh eval
项目 MUST 提供 `scripts/eval_night_grid.py` 对 manifest 中已完成 run 做 fresh apples-to-apples eval。该脚本 MUST 使用统一 checkpoint resolver 和统一 missing pattern helper，缺失 checkpoint MUST warning 但不中断。

#### Scenario: 输出 pattern metrics
- **WHEN** eval 脚本找到某 run checkpoint
- **THEN** 输出 `night_grid_metrics.csv`、`night_grid_metrics.md` 和 `checkpoint_manifest.json`
- **AND** CSV 行 MUST 包含 run、group、seed、pattern、Top-K、ADBA、MAE、loss、count、checkpoint path 和 checkpoint epoch

### Requirement: night grid analysis
项目 MUST 提供 `scripts/analyze_night_grid.py`，从 fresh eval 指标计算 by-run、by-group、mean/std、delta-vs-proto、top candidates 和 paper observations。排序 MUST 支持 balanced_score，并惩罚相对 proto 损伤 missing_gps、missing_radar 和 full top1 的候选。

#### Scenario: top candidates 输出
- **WHEN** analysis 脚本运行成功
- **THEN** `night_grid_top_candidates.md` MUST 列出 best avg_missing、best radar_only、best lidar_only、best balanced_score、best without hurting missing_gps、best without hurting missing_radar 和 seed3/40 epoch follow-up top3
- **AND** 若提升小于 seed std，报告 MUST 提示谨慎

### Requirement: summary 兼容 night grid
`scripts/summarize_missing_runs.py` MUST 支持 manifest 输入并识别 night grid run 状态。状态 MUST 至少包括 completed、completed_early_stopped、incomplete_has_checkpoint、killed_or_failed 和 missing。

#### Scenario: manifest summary 字段
- **WHEN** 用户传入 night grid manifest 和 expected epochs
- **THEN** summary 输出 MUST 至少包含 `run_name,group,status,best_epoch,final_epoch,best_val_acc,best_val_adba,best_checkpoint,log_path,exit_code`

### Requirement: Scene31 next-round local follow-up workflow
项目 MUST 将 Scene31 next-round follow-up 作为 local/manual experiment workflow 处理。该 workflow MUST 复用现有 `kd-sensing-train`、missing-pattern fresh eval 和本地输出边界，不得改变已有 Scene31 es20 night-grid 配置或 baseline 行为。

#### Scenario: next-round fresh eval 查找配置
- **WHEN** fresh eval 需要评估 next-round manifest 中的 run
- **THEN** 配置查找 MUST 优先使用 run 目录下的 `final_config.yaml` 或 `resolved_config.yaml`
- **AND** 手写 `configs/scene31/<run>.yaml` MAY 作为 legacy/local fallback

#### Scenario: local/manual 输出边界
- **WHEN** 用户运行 Scene31 next-round launcher 或汇总脚本
- **THEN** 训练、评估和汇总产物 MUST 写入 ignored 的 `outputs/` 或 `logs/` 下
- **AND** 系统 MUST 不提交 checkpoint、日志、fresh eval CSV 或训练输出

### Requirement: Verify workflow 不等同训练 workflow
项目的 verify、CI、lint 和 smoke workflow MUST 与真实训练/评估 workflow 保持边界清晰。Verify workflow 只能检查源码、配置、OpenSpec、CLI help、synthetic forward 或 mock schema；真实训练、长时间评估、feature cache 生成和 checkpoint 写入仍 MUST 通过显式训练/评估入口触发。

#### Scenario: CI 不启动真实训练
- **WHEN** CI 或 quick verify 在无真实数据环境中运行
- **THEN** 系统 MUST 不调用长时间 `kd-sensing-train` 真实训练
- **AND** 如需训练路径 smoke，MUST 使用 synthetic/mock fixture 或已有 focused test

#### Scenario: 训练仍使用 package CLI
- **WHEN** 用户需要运行真实实验
- **THEN** 文档 MUST 继续指向 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess` 或已登记 package CLI
- **AND** verify 入口 MUST 不成为新的长期训练入口

