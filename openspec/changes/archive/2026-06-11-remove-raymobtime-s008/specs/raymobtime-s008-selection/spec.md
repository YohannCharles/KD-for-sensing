## REMOVED Requirements

### Requirement: Raymobtime s008 数据审计与 cache 构建
**Reason**: Raymobtime s008 工作流已退役，项目不再维护其审计、index、ray feature、beam label 或 cache 构建能力。
**Migration**: 不提供 Raymobtime s008 替代入口；使用当前保留的 DeepSense6G、MMW、CSI 或 viewer workflow。

#### Scenario: Raymobtime 预处理器不存在
- **WHEN** 用户请求 Raymobtime s008 audit、index、ray feature 或 cache 预处理器
- **THEN** 系统 MUST 报告 Raymobtime s008 已退役或该 preprocessor 不存在
- **AND** 系统 MUST 不读取 `dataset/Raymobtime/s008`

### Requirement: Raymobtime s008 snapshot dataset 契约
**Reason**: `raymobtime_s008` dataset type 已退役，项目不再维护 current snapshot flat batch 输出契约。
**Migration**: 不提供兼容 dataset type；新实验必须基于当前保留数据集或另行提出 OpenSpec change。

#### Scenario: Raymobtime dataset type 被拒绝
- **WHEN** 用户配置 `data.dataset.type: raymobtime_s008`
- **THEN** dataset 构建 MUST 失败
- **AND** 错误信息 MUST 指出 Raymobtime s008 已退役

### Requirement: Raymobtime s008 current beam selection 模型
**Reason**: Raymobtime s008 专用 selection 模型和 LiDAR 3D occupancy encoder 已退役，不再作为 model registry 的当前支持项。
**Migration**: 不提供兼容模型别名；使用当前保留模型配置或为新模型创建独立 OpenSpec change。

#### Scenario: Raymobtime 模型注册被移除
- **WHEN** 用户请求 `simple_concat_multitask_selection`、`task_aware_gated_multitask_selection` 或 `raymobtime_lidar_3d_cnn` 作为 Raymobtime s008 模型
- **THEN** 模型构建 MUST 失败或报告该 Raymobtime s008 模型已退役
- **AND** 系统 MUST 不导入 `kd_sensing.models.raymobtime_s008`

### Requirement: Raymobtime s008 评估指标
**Reason**: Raymobtime s008 current beam、LOS 和 link quality objective 已退役，其专属正式指标不再需要写入训练/评估报告。
**Migration**: 当前保留 workflow 继续使用各自 objective 的指标；不迁移 Raymobtime 专属 metrics。

#### Scenario: Raymobtime 指标不再作为正式指标
- **WHEN** 用户加载旧 Raymobtime s008 objective 或评估配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不要求 `beam_dba_current`、`los_auc`、`link_rmse` 或 `selection_multitask_loss` 作为当前支持 workflow 的正式指标
