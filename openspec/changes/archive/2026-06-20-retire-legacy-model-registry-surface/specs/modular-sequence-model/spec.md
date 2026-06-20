## MODIFIED Requirements

### Requirement: 新入口不破坏保留模型
模块化序列模型 MUST 作为普通 supervised/adaptation baseline 的 canonical 组合入口存在。已迁移到 `modular_sequence` 的 legacy strong/lightweight 整模型注册名 MAY 被退役为 removed guard；仍保留的完整 `MODELS` 注册名 MUST 属于 current whole-model exception、workflow/paper reproduction 或明确 spec 需求。

#### Scenario: 退役注册名不可构建但可诊断
- **WHEN** 构建流程导入默认模型组件后请求已退役的 `image_strong`、`radar_lightweight`、`gps_strong`、`lidar_lightweight`、`mmwave_strong`、`fusion_lightweight` 或等价旧注册名
- **THEN** 系统 MUST 拒绝构建
- **AND** 错误信息 MUST 指向 `modular_sequence` 以及对应 encoder/core/head 迁移组合

#### Scenario: canonical 单模态配置仍可构建
- **WHEN** 用户加载 current image、radar、GPS、LiDAR 或 mmWave canonical root config
- **THEN** 系统 MUST 构建 `modular_sequence` 模型
- **AND** 模型 MUST 按启用模态解析 encoder、projector、representation core 和 beam head
- **AND** 训练循环 MUST 不需要为旧整模型名称新增专用 forward 分支

#### Scenario: 保留 whole-model exception 独立构建
- **WHEN** 用户配置 current whole-model exception，例如 `bev_fusion_2604`、`jepa_msac`、`gps_conditioned_jepa` 或其它仍在 current spec 中保留的注册名
- **THEN** 系统 MUST 继续通过 `MODELS` 构建该模型
- **AND** 该模型 MUST 保持其 documented forward/output/metadata 契约
