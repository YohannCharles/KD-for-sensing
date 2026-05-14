## MODIFIED Requirements

### Requirement: Fusion 配置选择 CRAF 模型
Fusion 配置 MUST 能显式选择 CRAF 或 CRAF baseline 模型，同时继续使用现有 `modalities` 字段描述参与融合的模态集合。CRAF、token transformer 和 early-concat fusion MUST 通过 canonical 配置路径区分，系统 MUST 不再保留 legacy 配置 alias 作为兼容入口。

#### Scenario: 配置 CRAF fusion
- **WHEN** 用户在 fusion 配置中设置 `model.student.type: craf_fusion`
- **THEN** 系统 MUST 使用 `model.student.modalities` 构建 CRAF 模型
- **AND** 系统 MUST 继续使用 `experiment.task: fusion` 的 batch 输入准备流程

#### Scenario: 配置 token transformer fusion
- **WHEN** 用户在 fusion 配置中设置 token-only transformer baseline 的注册名
- **THEN** 系统 MUST 使用同一模态集合构建不带 reliability gate 的 token fusion baseline

#### Scenario: early-concat fusion 显式运行
- **WHEN** 用户继续运行 canonical early-concat fusion 配置
- **THEN** 系统 MUST 保持 early-concat fusion 行为
- **AND** 系统 MUST 不隐式启用 CRAF 训练 loss 或 diagnostics

### Requirement: Fusion 模型公开类名表达 teacher/student 职责
Early-concat fusion teacher 和 student MUST 暴露职责明确的公开 Python 类名。`fusion_teacher` 注册名 MUST 构建 `FusionTeacherModalityNet`，`fusion_student` 注册名 MUST 构建 `FusionStudentModalityNet`。旧类名 `old fusion teacher class alias` 和 `old fusion student class alias` MUST 不再作为兼容 alias 导出。

#### Scenario: 构建 fusion teacher 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_teacher`
- **THEN** 系统 MUST 返回 `FusionTeacherModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_teacher` forward 输出契约

#### Scenario: 构建 fusion student 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_student`
- **THEN** 系统 MUST 返回 `FusionStudentModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_student` forward 输出契约

#### Scenario: 旧类名 alias 被拒绝
- **WHEN** 开发者导入 `old fusion teacher class alias` 或 `old fusion student class alias`
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向 `FusionTeacherModalityNet` 或 `FusionStudentModalityNet`

## REMOVED Requirements

### Requirement: Fusion legacy 入口兼容
**Reason**: legacy fusion 配置路径只是 canonical image+radar 配置的旧命名入口，继续保留会扩大配置和文档矩阵。
**Migration**: 使用 `configs/fusion/image_radar_*` 或 overlay/canonical 配置生成路径。

#### Scenario: image+radar legacy fusion 入口
- **WHEN** 用户运行 `fusion/no_kd.yaml`、`fusion/logits_kd.yaml` 或 `fusion/rkd.yaml`
- **THEN** 系统 MUST 拒绝旧路径或不再提供这些文件
- **AND** 错误信息或文档 MUST 指向对应 canonical image+radar 配置

#### Scenario: 既有 fusion 示例入口
- **WHEN** 用户运行只作为旧示例命名存在的 fusion 配置
- **THEN** 系统 MUST 拒绝旧路径或不再提供这些文件
- **AND** 文档 MUST 指向 canonical student no-KD、teacher no-KD、logits KD 或 RKD 配置名称

### Requirement: Legacy fusion 配置不变
**Reason**: teacher-prior、CRAF、G2D 和 early-concat 行为已经通过显式 canonical 配置区分，不需要 legacy 配置不变性保护。
**Migration**: 使用 canonical fusion 配置并在 `final_config.yaml` 中保存完整解析结果。

#### Scenario: legacy fusion 仍按 image+radar 运行
- **WHEN** 用户加载旧 legacy fusion 配置路径
- **THEN** 系统 MUST 拒绝旧路径或不再提供这些文件
- **AND** 系统 MUST 不把旧路径静默映射到 canonical 配置
