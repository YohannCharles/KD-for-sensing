## ADDED Requirements

### Requirement: 原代码兼容仅保留迁移说明
项目 MUST 不再为上游原代码入口、旧配置矩阵、旧 checkpoint fallback 或旧输入结构提供运行兼容。文档可以保留历史差异说明，但训练、评估、配置解析和 registry MUST 只支持当前 canonical 路线。

#### Scenario: 旧原代码兼容配置被拒绝
- **WHEN** 用户加载只为复现上游旧入口保留的配置路径或字段
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指向当前 canonical 配置或要求显式提供 checkpoint

#### Scenario: 历史说明不产生运行入口
- **WHEN** 开发者阅读 README 或扩展指南中的历史差异说明
- **THEN** 文档 MUST 不推荐旧脚本、旧 config alias 或旧权重 fallback 作为可运行入口
- **AND** 文档 MUST 给出当前 canonical 训练和评估路线

## REMOVED Requirements

### Requirement: 原代码兼容与单模态一致性配置矩阵
**Reason**: 该矩阵要求保留上游旧脚本、旧超参数来源和 legacy 权重 fallback，已经与项目当前 canonical 配置矩阵冲突。
**Migration**: 使用当前 `configs/` canonical 配置、显式 scene 配置和 checkpoint registry；需要复现实验时保存完整 `final_config.yaml`。

#### Scenario: 单模态 GRU 层数对齐
- **WHEN** 用户依赖该旧兼容矩阵推导模型结构
- **THEN** 系统 MUST 不再保证旧矩阵语义
- **AND** 当前配置解析结果 MUST 以 canonical 配置文件和模型注册名为准

#### Scenario: teacher 权重默认路径可解析
- **WHEN** 用户运行 KD 配置且未提供 registry checkpoint 或显式 checkpoint
- **THEN** 系统 MUST 不再从 `All_models` 或 legacy fallback 路径静默加载权重
- **AND** 错误信息 MUST 要求训练/归档 teacher 或显式配置 checkpoint

### Requirement: 目标兼容固定输入尺寸约束
**Reason**: 旧“目标兼容”概念绑定上游兼容路径；当前尺寸约束应由具体 model/profile 契约表达。
**Migration**: 使用 image profile、radar preprocessing 和模型构建校验中的 canonical 输入契约。

#### Scenario: image size 不兼容
- **WHEN** 用户配置不被当前 image profile 或模型支持的 image size
- **THEN** 系统 MUST 通过当前 image profile 或模型校验拒绝该配置
- **AND** 错误信息 MUST 不引用已删除的目标兼容路径
