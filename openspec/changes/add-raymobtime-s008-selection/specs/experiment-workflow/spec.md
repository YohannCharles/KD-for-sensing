## ADDED Requirements

### Requirement: Raymobtime s008 配置驱动预处理 workflow
项目 MUST 为 Raymobtime s008 提供配置驱动的预处理 workflow。所有项目相关 Python 命令 MUST 使用 `conda run -n kd_mm_beam`，并 MUST 通过包内 CLI、`scripts/preprocess.py` 或 console script 调用。

#### Scenario: 运行 Raymobtime 审计命令
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/raymobtime_s008_audit.yaml`
- **THEN** 系统 MUST 执行 Raymobtime s008 审计预处理
- **AND** 命令输出 MUST 记录审计摘要路径

#### Scenario: 运行 Raymobtime cache 构建命令
- **WHEN** 用户运行 Raymobtime s008 index、ray feature 或 cache 构建配置
- **THEN** 系统 MUST 根据配置中的 `data_root`、`cache_dir`、split seed 和输出目录生成对应 cache
- **AND** 命令输出 MUST 记录 index、label、ray feature 和 split metadata 路径

#### Scenario: 预处理 action 可发现
- **WHEN** 用户查看 `conda run -n kd_mm_beam kd-sensing-preprocess --help`
- **THEN** 帮助信息 MUST 能发现 Raymobtime s008 预处理 action 或说明可通过配置中的 `preprocessing.type` 选择
- **AND** Raymobtime action MUST 不要求新增顶层旧脚本

### Requirement: Raymobtime s008 训练与评估 workflow
项目 MUST 提供可通过统一训练和评估入口运行的 Raymobtime s008 current beam selection 与 selection multitask 配置。训练和评估产物 MUST 记录 snapshot 语义、数据 split、启用模态、objective、指标和输出路径。

#### Scenario: 加载 Raymobtime multitask 配置
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-train --config configs/raymobtime/s008_multitask_selection.yaml`
- **THEN** 系统 MUST 构建 `raymobtime_s008` dataset
- **AND** 系统 MUST 构建 `selection_multitask` objective
- **AND** 系统 MUST 构建配置指定的 snapshot 多任务模型
- **AND** 系统 MUST 不加载 teacher checkpoint

#### Scenario: Raymobtime 评估报告
- **WHEN** 用户对 Raymobtime s008 checkpoint 运行统一评估入口
- **THEN** 评估报告 MUST 包含 beam Top-K、LOS 指标、link 指标、enabled modalities、objective、split metadata 路径和样本数
- **AND** 报告 MUST 标记该任务为 current snapshot beam selection

#### Scenario: 缺失 cache 的错误提示
- **WHEN** 用户在未构建 Raymobtime cache 的情况下启动训练
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 提示先运行 Raymobtime s008 审计、index、ray feature 和 cache 预处理

### Requirement: Raymobtime s008 smoke workflow
项目 MUST 提供不依赖真实大规模训练的 Raymobtime s008 smoke workflow，用于验证 dataset、模型、loss、metrics、checkpoint 和评估路径。smoke workflow MUST 使用小 fixture 或可配置的极小 batch 限制。

#### Scenario: Raymobtime dataset smoke test
- **WHEN** 开发者运行 Raymobtime s008 dataset 相关测试
- **THEN** 测试 MUST 使用小 fixture 验证审计、index、label 标准化、ray feature no-LOS 输入和 dataset sample contract
- **AND** 测试命令 MUST 使用 `conda run -n kd_mm_beam pytest ...`

#### Scenario: Raymobtime 训练 smoke test
- **WHEN** 开发者运行 Raymobtime s008 最小训练 smoke
- **THEN** 训练流程 MUST 完成 forward、loss、backward、validation 和 checkpoint 保存
- **AND** smoke 配置 MUST 使用 current snapshot 输出，不得创建 GRU/RNN/LSTM 模块

#### Scenario: Raymobtime 评估 smoke test
- **WHEN** 开发者运行 Raymobtime s008 最小评估 smoke
- **THEN** 评估流程 MUST 输出 beam、LOS 和 link 指标
- **AND** 评估报告 MUST 能被模态失衡分析读取

### Requirement: Raymobtime s008 实验矩阵与分析 workflow
项目 MUST 提供 Raymobtime s008 实验矩阵和分析 workflow，用于比较单模态、多模态、sensing-only、sensing+ray 和 task-aware gate 结果。分析 workflow MUST 通过包内 CLI 或统一分析入口执行。

#### Scenario: 单模态与多模态矩阵
- **WHEN** 用户查看 Raymobtime s008 推荐配置或分析说明
- **THEN** 系统 MUST 覆盖 `coord` only、`image` only、`lidar` only、`ray` only，以及至少 `coord+image`、`coord+lidar`、`coord+ray`、`image+lidar`、`coord+image+lidar` 和 `coord+image+lidar+ray`
- **AND** 每组 MUST 能选择 simple concat 或 task-aware gated 模型

#### Scenario: sensing-only 单任务主矩阵
- **WHEN** 用户需要 Raymobtime s008 sensing-only 单任务主实验
- **THEN** 推荐运行矩阵 MUST 覆盖 `coord`、`image`、`lidar` 和 `coord+image+lidar` 四组输入条件
- **AND** 每组 MUST 分别运行 `current_beam_selection`、`current_los_classification` 和 `current_link_quality`
- **AND** 该矩阵共 12 个训练 run，包含 `ray` 的 sensing+ray run MUST 单独标注为补充实验

#### Scenario: 分析命令输出
- **WHEN** 用户运行 Raymobtime s008 模态失衡分析入口
- **THEN** 系统 MUST 读取一个或多个 Raymobtime 实验目录
- **AND** 系统 MUST 输出 CSV 或 JSON 摘要，包含单模态性能、gate 均值、modality drop delta 和按 LOS bucket 的任务指标

#### Scenario: 分析元数据校验
- **WHEN** 分析入口读取实验目录
- **THEN** 系统 MUST 校验 run metadata 中的 dataset type、objective、enabled modalities 和 task semantics
- **AND** 非 Raymobtime 或 future beam prediction 实验 MUST 不被静默纳入 Raymobtime s008 汇总
