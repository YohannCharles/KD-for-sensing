## MODIFIED Requirements

### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、`model.primary` 主模型、supervised/adaptation/JEPA/BGAM/CSI 或诊断目标、训练超参数、优化器、调度器、输出目录、随机种子、GPS 特征模式和 fusion 模态选择。当前支持的训练配置 MUST 不覆盖 KD 模式或 teacher checkpoint；旧 KD、teacher/student no-KD、Hist、Top8 standalone、residual 和 camera residual 路径 MUST 在配置解析或 registry 层被拒绝。

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

#### Scenario: 使用当前 JEPA 和 BGAM workflow
- **WHEN** 用户运行当前 JEPA pretraining/downstream、GPS-query pooling、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、viewer manifest 或 benchmark 配置
- **THEN** 系统 MUST 使用对应 current workflow 的 `model.primary`、runner manifest 或诊断 schema
- **AND** 系统 MUST 不恢复 legacy KD、Hist、standalone Top8 selector、GPS residual 或 camera residual runtime

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
- **AND** 错误信息 MUST 指向当前 `model.primary`、supervised/adaptation、JEPA、BGAM 或保留 baseline 入口

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

### Requirement: 默认实验入口去 KD-first 化
项目默认 quickstart、README 推荐入口、当前主线 quick validation 和新 canonical mainline 配置 MUST 以 supervised/adaptation、JEPA、BGAM、CSI hardening、baseline/control、诊断或 viewer manifest 工作流为默认。旧 KD 配置不得作为当前主线默认实验入口。

#### Scenario: README quickstart 使用当前主线
- **WHEN** 开发者阅读 README 或当前主线运行说明
- **THEN** 推荐的首个训练、评估或诊断命令 MUST 使用当前 supervised/adaptation、JEPA、BGAM、CSI、baseline/control 或 viewer manifest 配置
- **AND** 文档 MUST 不把 `logits_kd`、`rkd`、Hist/HiST、standalone Top8 selector、GPS residual 或 camera residual 作为当前主线 quickstart

#### Scenario: canonical mainline 配置不要求 teacher checkpoint
- **WHEN** 用户加载当前推荐的 mainline 配置
- **THEN** 配置 MUST 能在没有 teacher checkpoint 的情况下完成解析和 dry-run/smoke 构建
- **AND** 输出 metadata MUST 不记录 KD-enabled lineage

### Requirement: 当前推荐 workflow 聚焦少样本跨场景主线
README、实验矩阵和 quickstart MUST 将当前推荐 workflow 聚焦于 supervised/adaptation baseline、Image+GPS JEPA query-pool、paired baseline/control、Vision-Position baseline suite、Arnold22 Camera AE+GPS Direct、DeepSense6G/MMW BGAM、MMW GPS v2、CSI hardening、JEPA visual analysis、GPS shortcut benchmark、预处理、诊断和 viewer manifest。KD baseline、HiST-Beam/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS coarse anchor、residual fusion、camera residual、模态失衡诊断脚本、objective-aware auxiliary tasks 和 snapshot next-frame MUST 作为 optional、supporting、historical 或 retired workflow 描述，不得作为 few-shot cross-scene 默认主线步骤。

#### Scenario: quickstart 不推荐退役脚本
- **WHEN** 开发者阅读 README 或 `docs/experiment_matrix.md`
- **THEN** 文档 MUST 不推荐运行 `kd-sensing-hist-beam-loso`、`configs/hist_beam/*`、Raymobtime s008、retired Top8 selector/residual/GPS coarse anchor 命令或已退役的独立模态诊断脚本
- **AND** 若需要当前主线实验，文档 MUST 指向仍存在的配置化 CLI 或包内 workflow

#### Scenario: optional workflow 与主线区分
- **WHEN** 文档提到 legacy KD、HiST-Beam、Top8 selector、residual、camera residual、GPS coarse anchor、snapshot next-frame、occlusion、position 或 multitask objective
- **THEN** 文档 MUST 明确它们不是当前主结论的默认步骤
- **AND** 文档 MUST 不要求先运行这些支线才能执行当前 DeepSense6G/MMW/JEPA/BGAM/CSI 主线

#### Scenario: 当前 workflow 文档声明运行状态
- **WHEN** 文档列出当前实验配置、benchmark manifest 或诊断配置
- **THEN** 文档 MUST 标明该条目是 formal、lowmem、smoke、debug、evaluation-only、upper-bound、historical ablation 还是 mock
- **AND** upper-bound、mock、smoke 或 historical ablation MUST 不得被写成正式结论

## REMOVED Requirements

### Requirement: RadarStudent no-KD 实验配置
**Reason**: 当前项目已将旧 no-KD/teacher/student 路径收敛为 `model.primary`、strong/lightweight/supervised 和 migration guard；`configs/radar/student_no_kd.yaml` 不再作为当前推荐入口维护。

**Migration**: 使用 `configs/radar/lightweight.yaml`、`configs/radar/supervised.yaml` 或其它当前 radar `model.primary` 配置；旧 `student_no_kd` 请求应由配置加载器拒绝并提示迁移。

#### Scenario: 旧 RadarStudent no-KD 请求迁移
- **WHEN** 用户请求运行 `configs/radar/student_no_kd.yaml`
- **THEN** 系统 MUST 拒绝该旧入口
- **AND** 错误信息 MUST 指向当前 radar lightweight 或 supervised 配置
