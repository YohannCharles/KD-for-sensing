## REMOVED Requirements

### Requirement: Raymobtime 数据集家族目录规范
**Reason**: Raymobtime s008 已退役，项目不再提供 `dataset/Raymobtime/s008` 默认数据根目录或 Raymobtime layout descriptor。
**Migration**: 不提供 Raymobtime s008 目录迁移；当前保留数据集继续使用各自规范目录。

#### Scenario: Raymobtime 默认目录不再解析
- **WHEN** 代码请求 Raymobtime s008 默认数据根目录
- **THEN** 系统 MUST 报告 Raymobtime s008 已退役或 dataset type 不存在
- **AND** 系统 MUST 不返回 `dataset/Raymobtime/s008` 作为可运行默认路径

### Requirement: Raymobtime s008 本地产物边界
**Reason**: Raymobtime s008 运行产物契约随工作流退役而删除。
**Migration**: 删除本地 Raymobtime s008 数据和产物时使用 `raymobtime-s008-retirement` 的 manifest 清理要求。

#### Scenario: Raymobtime 产物不再作为当前支持输出
- **WHEN** 用户查看当前数据集目录和产物说明
- **THEN** 文档 MUST 不再要求 Raymobtime s008 cache、审计报告、训练输出或 checkpoint 作为当前 workflow 产物
- **AND** 历史产物清理 MUST 遵守 manifest 边界
