# deepsense6g-scene-selection Specification

## Purpose
定义 DeepSense6G 场景选择、数据根目录解析和跨场景输出隔离规则，确保训练、评估和诊断在多场景间保持可复现。
## Requirements
### Requirement: DeepSense6G 场景选择配置
项目 MUST 支持通过配置选择 DeepSense6G 场景。`data.dataset.type` MUST 使用 `deepsense6g`，`data.dataset.scene` MUST 接受整数和字符串别名，当前 MUST 支持 Scenario 9、Scenario 31、Scenario 32、Scenario 33 与 Scenario 34。未显式设置场景时，通用 DeepSense6G 配置 MUST 默认使用 Scenario 31。旧 `the scene-9 dataset-type spelling`、`scenario31`、`scenario32`、`scenario33` 和 `scenario34` dataset type 配置 MUST 被拒绝并给出迁移提示。

#### Scenario: 默认使用 Scenario 31
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 训练配置
- **THEN** 系统 MUST 将场景解析为 Scenario 31
- **AND** 默认数据根目录 MUST 指向 Scenario 31 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 31` 和 `scene_slug: scene31`

#### Scenario: 通过整数选择 Scenario 9
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 9`
- **THEN** 系统 MUST 将场景解析为 Scenario 9
- **AND** 默认数据根目录 MUST 指向 Scenario 9 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 通过整数选择 Scenario 32
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 系统 MUST 将场景解析为 Scenario 32
- **AND** 默认数据根目录 MUST 指向 Scenario 32 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 32` 和 `scene_slug: scene32`

#### Scenario: 通过整数选择 Scenario 33
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 33`
- **THEN** 系统 MUST 将场景解析为 Scenario 33
- **AND** 默认数据根目录 MUST 指向 Scenario 33 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 33` 和 `scene_slug: scene33`

#### Scenario: 通过整数选择 Scenario 34
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 34`
- **THEN** 系统 MUST 将场景解析为 Scenario 34
- **AND** 默认数据根目录 MUST 指向 Scenario 34 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 34` 和 `scene_slug: scene34`

#### Scenario: 通过别名选择场景
- **WHEN** 用户设置 `data.dataset.scene` 为 `scene9`、`scenario9`、`scene31`、`scenario31`、`scene32`、`scenario32`、`scene33`、`scenario33`、`scene34` 或 `scenario34`
- **THEN** 系统 MUST 解析到对应的规范场景编号
- **AND** 配置中的大小写差异 MUST 不影响解析结果

#### Scenario: 旧 dataset type 被拒绝
- **WHEN** 用户设置 `the scene-9 dataset-type spelling`、`scenario31`、`scenario32`、`scenario33` 或 `scenario34` 作为 `data.dataset.type`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和对应 `data.dataset.scene`

#### Scenario: 未知场景被拒绝
- **WHEN** 用户设置未注册的 `data.dataset.scene`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 列出当前支持的场景

### Requirement: 场景默认路径和显式覆盖
DeepSense6G 场景解析 MUST 为每个支持场景提供默认数据根目录、legacy 数据根目录、train CSV 名和 test CSV 名。未显式配置 `data_root` 时，默认数据根目录 MUST 使用 `dataset/DeepSense6G/scenario*` 家族目录。用户显式配置的 `data_root`、`train_csv_name` 或 `test_csv_name` MUST 覆盖场景默认值。

#### Scenario: 使用默认 Scenario 31 路径
- **WHEN** 用户选择 Scenario 31 或未显式设置 `data.dataset.scene`，且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/DeepSense6G/scenario31` 作为 Scenario 31 的默认数据根目录
- **AND** train/test dataset MUST 使用该场景的默认 split CSV 名

#### Scenario: 使用默认 Scenario 33 路径
- **WHEN** 用户选择 Scenario 33，且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/DeepSense6G/scenario33` 作为 Scenario 33 的默认数据根目录
- **AND** train/test dataset MUST 使用该场景的默认 split CSV 名

#### Scenario: 使用默认 Scenario 34 路径
- **WHEN** 用户选择 Scenario 34，且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/DeepSense6G/scenario34` 作为 Scenario 34 的默认数据根目录
- **AND** train/test dataset MUST 使用该场景的默认 split CSV 名

#### Scenario: 显式 data_root 覆盖默认值
- **WHEN** 用户选择 Scenario 31 并设置 `data.dataset.data_root: /tmp/custom_scene31`
- **THEN** 系统 MUST 使用 `/tmp/custom_scene31` 构建 dataset
- **AND** 系统 MUST 仍在 metadata 中记录规范场景为 Scenario 31

#### Scenario: 显式旧 data_root 兼容
- **WHEN** 用户选择 Scenario 31 并设置 `data.dataset.data_root: dataset/scenario31`
- **THEN** 系统 MUST 使用 `dataset/scenario31` 构建 dataset
- **AND** 系统 MUST 不把该显式路径改写为 `dataset/DeepSense6G/scenario31`

#### Scenario: 显式 CSV 覆盖默认值
- **WHEN** 用户设置 `data.dataset.train_csv_name` 或 `data.dataset.test_csv_name`
- **THEN** 系统 MUST 使用显式 CSV 名构建对应 split
- **AND** 场景默认 CSV 名不得覆盖用户显式设置

### Requirement: 场景扩展约定
项目 MUST 提供可维护的场景注册或描述符机制，使未来新增 DeepSense6G 场景不需要复制数据集类或重写训练入口。

#### Scenario: 新增场景描述符
- **WHEN** 开发者新增一个包含场景编号、slug、别名和默认路径的场景描述符
- **THEN** 配置解析、dataset 构建、输出目录分组和 metadata 记录 MUST 能复用同一描述符
- **AND** 新增场景不得要求复制 Scenario 9 数据集读取逻辑

### Requirement: DeepSense6G 规范目录清单
DeepSense6G MUST 将当前支持场景的规范数据根目录定义为 `dataset/DeepSense6G/scenario9`、`dataset/DeepSense6G/scenario31`、`dataset/DeepSense6G/scenario32`、`dataset/DeepSense6G/scenario33` 和 `dataset/DeepSense6G/scenario34`。这些默认路径 MUST 由同一场景描述符或 dataset layout descriptor 提供。

#### Scenario: Scenario 9 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 9`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario9`

#### Scenario: Scenario 31 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 31`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario31`

#### Scenario: Scenario 32 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario32`

#### Scenario: Scenario 33 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 33`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario33`

#### Scenario: Scenario 34 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 34`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario34`

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

### Requirement: DeepSense6G 场景感知数据构建
数据构建流程 MUST 根据 DeepSense6G 场景选择和 dataset layout descriptor 解析数据根目录和 split CSV。canonical 配置 MUST 使用 `data.dataset.type: deepsense6g`。旧 `scenario9`、`scenario31` 和 `scenario32` dataset type MUST 不再可构建。

#### Scenario: 旧 scenario9 配置被拒绝
- **WHEN** 用户运行包含 `the scene-9 dataset-type spelling` 的旧配置
- **THEN** 数据构建流程 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 通用 deepsense6g 类型默认选择 Scenario 31
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且未显式设置 `data.dataset.scene`
- **THEN** 数据构建流程 MUST 构建 Scenario 31 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario31`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: 显式选择 Scenario 32
- **WHEN** 用户运行 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 数据构建流程 MUST 构建 Scenario 32 对应的 DeepSense6G dataset
- **AND** 数据根目录 MUST 默认为 `dataset/DeepSense6G/scenario32`
- **AND** 启用模态推导 MUST 继续按 `experiment.task` 或 fusion `modalities` 生效

#### Scenario: split metadata 记录场景
- **WHEN** 训练或评估构建 train/test dataset
- **THEN** split metadata MUST 记录每个 split 的 `scene_id`、`scene_slug`、CSV 路径和样本数
- **AND** 这些字段 MUST 出现在最终配置、运行日志或测试报告中

#### Scenario: 场景不影响模态按需读取
- **WHEN** 用户在任一受支持 DeepSense6G 场景上运行 mmWave-only 或 GPS+mmWave fusion 配置
- **THEN** dataset MUST 只读取启用模态所需文件和 beam label 文件
- **AND** 未启用模态的缺失文件不得阻止该任务运行

### Requirement: DeepSense6G 预处理路径重定向
DeepSense6G 序列 CSV 预处理 MUST 使用 dataset layout descriptor 解析 scene root。命令行或配置中的场景覆盖 MUST 同时更新 `preprocessing.data_root` 和默认 `preprocessing.csv_path` 到目标场景的规范目录，除非用户显式提供自定义绝对路径。

#### Scenario: 预处理默认 Scenario 31 路径
- **WHEN** 用户运行默认 DeepSense6G sequence CSV 预处理配置
- **THEN** `preprocessing.data_root` MUST 指向 `dataset/DeepSense6G/scenario31`
- **AND** `preprocessing.csv_path` MUST 指向 `dataset/DeepSense6G/scenario31/scenario31_RA.csv`

#### Scenario: 预处理场景覆盖到 Scenario 9
- **WHEN** 用户在 sequence CSV 预处理中覆盖 `data.dataset.scene: 9`
- **THEN** `preprocessing.data_root` MUST 更新为 `dataset/DeepSense6G/scenario9`
- **AND** 默认 `preprocessing.csv_path` MUST 更新为 `dataset/DeepSense6G/scenario9/scenario9_RA.csv`

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
