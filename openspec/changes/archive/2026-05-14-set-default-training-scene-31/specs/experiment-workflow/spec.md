## MODIFIED Requirements

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
