## REMOVED Requirements

### Requirement: Multimodal-NF 数据集家族目录规范
**Reason**: Multimodal-NF 数据集家族退役，项目不再为其维护默认目录、layout descriptor 或路径覆盖语义。
**Migration**: 本地已有 `dataset/MultimodalNF/` 可作为静态用户文件保留；当前项目不会自动移动、读取或审计该目录。

#### Scenario: Multimodal-NF 默认目录删除
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** dataset layout descriptor MUST 不再返回 `dataset/MultimodalNF`
- **AND** dataset 构建 MUST 失败或报告该 dataset type 已不受支持

### Requirement: Multimodal-NF 本地产物边界
**Reason**: Multimodal-NF 训练、审计、cache 和 checkpoint 工作流退役。
**Migration**: 通用本地产物边界仍适用于当前保留数据集和训练输出；历史 Multimodal-NF 本地产物不纳入源码变更。

#### Scenario: Multimodal-NF 产物不再由项目生成
- **WHEN** 用户运行当前保留预处理、训练或评估命令
- **THEN** 系统 MUST 不生成 Multimodal-NF cache、审计报告或训练输出

### Requirement: Multimodal-NF 数据文件不自动迁移
**Reason**: 项目不再管理 Multimodal-NF 数据文件，因此不再需要专属迁移约束。
**Migration**: 实现本 change 时仍不得主动删除、移动或解压用户本地 `dataset/MultimodalNF/`；后续项目运行不再消费该路径。

#### Scenario: 实施删除不触碰真实数据
- **WHEN** 开发者实现本 change
- **THEN** 任务 MUST 不包含删除、移动或解压 `dataset/MultimodalNF/` 下的真实数据
- **AND** 只删除源码、配置、测试和文档中的支持入口
