## ADDED Requirements

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
