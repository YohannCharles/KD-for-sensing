## ADDED Requirements

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

## REMOVED Requirements

### Requirement: 当前训练产物迁移到 Scenario 9
**Reason**: 该要求把所有 `outputs/<run_name>/` 历史训练目录统一归入 Scenario 9，无法表达当前主线的 scene31、S32-S34、S31-S34 和 BeamBench/Arnold22 多场景运行，容易把多场景 JEPA/fusion run 误标为单场景产物。

**Migration**: 使用 runtime output organize manifest 对历史产物逐项分类。单场景 run 迁入对应 `outputs/scene<id>/`；多场景 run 迁入 `outputs/scenegroup_<scene-range-or-list>/`；无法可靠判定 scope 或仍有旧路径引用的产物进入 `outputs/archive/` 或人工复核列表。

## MODIFIED Requirements

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
