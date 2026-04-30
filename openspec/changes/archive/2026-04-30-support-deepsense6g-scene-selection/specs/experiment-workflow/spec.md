## ADDED Requirements

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
