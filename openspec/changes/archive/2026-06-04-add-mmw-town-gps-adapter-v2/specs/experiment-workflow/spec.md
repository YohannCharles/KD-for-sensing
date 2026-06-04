## ADDED Requirements

### Requirement: MMW Town GPS v2 CLI workflow
项目 MUST 提供配置驱动的 MMW Town GPS-only v2 runner、plotter 和 comparison 入口。入口 MUST 位于 `kd_sensing` 包内并可通过 console script 或 `python -m kd_sensing.cli.<module>` 运行；项目 MUST NOT 要求用户通过 `python -m src.*` 调用该 workflow。

#### Scenario: v2 runner help 可用
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-mmw-town-gps-v2 --help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 `--config`、`--label-space`、`--target-scene`、`--support-ratio`、`--support-num` 和 `--support-mode`

#### Scenario: plotter 和 comparison help 可用
- **WHEN** 用户执行 v2 plotter 或 comparison console script 的 `--help`
- **THEN** 命令 MUST 正常退出
- **AND** 帮助信息 MUST 包含 results dir、previous dir 或 new dir 等必要参数

### Requirement: MMW Town GPS v2 default configuration
项目 MUST 提供 `configs/mmw_town_gps_adapter_v2.yaml` 或等价 v2 配置。配置 MUST 声明数据根、已有分析目录、label space、四个 scene、num_beams、split、model、loss、train、adapt、metrics 和 ablation 矩阵。

#### Scenario: 默认配置可解析
- **WHEN** 用户通过 v2 runner 传入默认 v2 配置
- **THEN** 系统 MUST 能解析完整配置
- **AND** 默认 label space MUST 为 `mapping_enabled`
- **AND** 默认 scene 列表 MUST 覆盖 crossroad、skybridge、curvyroad 和 Hroad

### Requirement: README documents MMW Town GPS v2
README MUST 增加 MMW Town GPS-only v2 说明，覆盖普通跨场景 GPS 分类器失败原因、circular beam distance、mapping_enabled/mapping_disabled、SceneAdapterV2 三种 adapter、完整实验命令、summary_by_scene 解读、crossroad/Hroad 残差诊断和后续多模态 residual correction 边界。

#### Scenario: README 提供可执行命令
- **WHEN** 开发者阅读 README 的 MMW Town GPS-only v2 小节
- **THEN** 文档 MUST 提供使用 `conda run -n kd_mm_beam` 的 runner、plotter 和 comparison 命令
- **AND** 文档 MUST 明确本 change 不实现多模态 residual correction
