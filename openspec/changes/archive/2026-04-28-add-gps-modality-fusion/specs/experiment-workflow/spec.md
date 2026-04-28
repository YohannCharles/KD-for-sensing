## ADDED Requirements

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

## MODIFIED Requirements

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
