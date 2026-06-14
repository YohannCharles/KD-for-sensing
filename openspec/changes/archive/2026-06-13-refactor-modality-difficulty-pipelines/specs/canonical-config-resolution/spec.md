## ADDED Requirements

### Requirement: Difficulty 配置解析与校验
配置加载流程 MUST 支持解析 top-level 或等价位置的 difficulty profiles，并在实体 YAML、virtual canonical 配置和命令行覆盖之后执行标准化与校验。解析结果 MUST 包含 profile id、operator list、stage/split selector、condition、severity、seed、affected modalities、fallback 和 digest。未知 operator、未知模态、非法 stage/split、非法 severity 或 target-shift 配置 MUST 被拒绝。

#### Scenario: 实体配置解析 difficulty profiles
- **WHEN** 用户加载包含 difficulty profiles 的实体 YAML
- **THEN** 配置加载流程 MUST 在 defaults、overlay 和命令行覆盖后标准化 difficulty 配置
- **AND** `final_config.yaml` 或 resolved config MUST 记录标准化后的 profile 和 digest

#### Scenario: 命令行覆盖 difficulty severity
- **WHEN** 用户通过命令行覆盖某个 difficulty profile 的 severity
- **THEN** 覆盖 MUST 在 difficulty validation 前生效
- **AND** resolved profile digest MUST 反映覆盖后的参数

#### Scenario: 非法 stage 被拒绝
- **WHEN** 配置声明 difficulty stage `preprocess_dataset_files`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 列出允许的 stage，例如 train、validation、test、evaluation 和 benchmark

### Requirement: Difficulty overlay recipe
canonical/virtual config recipe MAY 生成当前支持的 difficulty overlay，例如 clean、GPS mild async、GPS severe async、GPS/image dropout training、image hard degradation 和 benchmark sweep。recipe 生成的 difficulty 配置 MUST 与实体 YAML 使用同一标准化、validation 和 digest 流程。已退役的旧 KD、G2D、CRAF、MARF 或 image motion profile MUST 不得通过 difficulty overlay 恢复。

#### Scenario: 生成 mild async training overlay
- **WHEN** 用户加载声明支持的 mild async training overlay 配置
- **THEN** 系统 MUST 生成 train-stage GPS async difficulty profile
- **AND** 其它 supervised/fusion 配置语义 MUST 继续由当前 canonical recipe 决定

#### Scenario: difficulty overlay 不恢复 image motion profile
- **WHEN** 用户加载 image degradation difficulty overlay
- **THEN** image modality input profile MUST 仍解析为当前 RGB/ImageNet profile
- **AND** 系统 MUST 不生成或接受已删除的 `motion_mask` image profile、image motion cache 或 motion encoder

### Requirement: Difficulty 解析产物可比较
系统 MUST 为 difficulty profiles 生成稳定 digest，用于 run metadata、benchmark comparability 和论文图表分组。digest MUST 基于标准化后的 operator、condition、severity、stage/split、seed 和 fallback，而不是用户 YAML 字段顺序。

#### Scenario: 字段顺序不同 digest 相同
- **WHEN** 两个配置声明语义相同但 YAML 字段顺序不同的 difficulty profile
- **THEN** 标准化后的 digest MUST 相同
- **AND** benchmark comparability MAY 将它们视为同一 difficulty condition

#### Scenario: severity 改变 digest 改变
- **WHEN** 两个 profile 仅 severity 不同
- **THEN** digest MUST 不同
- **AND** 输出指标 MUST 能按不同 severity 分组
