## ADDED Requirements

### Requirement: Fusion 模型公开类名表达 teacher/student 职责
Legacy early-concat fusion teacher 和 student MUST 暴露职责明确的公开 Python 类名。`fusion_teacher` 注册名 MUST 构建 `FusionTeacherModalityNet`，`fusion_student` 注册名 MUST 构建 `FusionStudentModalityNet`。旧类名 `FusionModalityNet` 和 `StudentModalityNet` MAY 作为兼容 alias 保留，但新代码、文档和测试 MUST 优先使用新类名。

#### Scenario: 构建 fusion teacher 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_teacher`
- **THEN** 系统 MUST 返回 `FusionTeacherModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_teacher` forward 输出契约

#### Scenario: 构建 fusion student 返回新类名
- **WHEN** 开发者通过 `MODELS.build()` 构建 `type: fusion_student`
- **THEN** 系统 MUST 返回 `FusionStudentModalityNet` 实例
- **AND** 该实例 MUST 保持既有 `fusion_student` forward 输出契约

#### Scenario: 旧类名作为兼容 alias
- **WHEN** 现有代码从 `kd_sensing.models.fusion` 导入 `FusionModalityNet` 或 `StudentModalityNet`
- **THEN** 导入 MUST 继续成功
- **AND** 旧类名 MUST 指向对应的新 fusion teacher/student 实现
