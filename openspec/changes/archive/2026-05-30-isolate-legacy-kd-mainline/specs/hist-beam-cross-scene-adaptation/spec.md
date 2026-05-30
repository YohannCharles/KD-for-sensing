## ADDED Requirements

### Requirement: HiST-Beam 主线默认 no-KD
HiST-Beam 跨场景适配主线 MUST 默认使用 no-KD supervised/adaptation 训练。source-only、shared/private、adapter-only、adapter+prototype、path/radio prototype、history-anchored residual、private calibration 和 full fine-tuning baseline MUST 不要求 teacher-student distillation、teacher checkpoint 或 KD loss。

#### Scenario: HiST-Beam source training 不加载 teacher
- **WHEN** 用户运行当前推荐的 HiST-Beam source training 或 sensor-assisted LOSO source stage
- **THEN** 系统 MUST 只构建当前配置指定的主模型
- **AND** 系统 MUST 不构建 frozen teacher model
- **AND** run metadata MUST 记录 `distillation_enabled=false`

#### Scenario: target adaptation 不计算 KD loss
- **WHEN** 用户运行 adapter、prototype、path/radio prototype、residual calibration 或 full fine-tuning target adaptation
- **THEN** adaptation loss MUST 来自 supervised target loss、无标签一致性/prototype/entropy/calibration loss 或对应方法定义
- **AND** adaptation MUST 不要求 teacher/student logits 对齐

### Requirement: KD 仅作为显式 HiST-Beam baseline 或增强
HiST-Beam 工作流 MAY 保留 KD baseline 或后续 KD 增强，但必须显式 opt-in，并且 MUST 与默认 HiST-Beam 主线矩阵分离。KD baseline 不得静默加入 sensor-assisted quick validation、history-anchored residual quick validation 或主结论 conclusion。

#### Scenario: 显式运行 HiST-Beam KD baseline
- **WHEN** 用户选择明确标注为 HiST-Beam KD baseline 的配置
- **THEN** plan metadata MUST 记录 `method_family=legacy_kd` 或等价字段
- **AND** run metadata MUST 记录 teacher source、teacher checkpoint、distillation type 和 student variant
- **AND** summary MUST 将该 run 归入 KD baseline 分组

#### Scenario: 默认矩阵排除 KD baseline
- **WHEN** 用户生成默认 HiST-Beam quick validation、MMW sensor-assisted quick validation 或 history-anchored residual quick validation plan
- **THEN** plan MUST 不包含 KD baseline variant
- **AND** 只有用户显式选择 legacy KD baseline profile 时才生成 KD run

### Requirement: HiST-Beam shared/private 语义不依赖 KD
HiST-Beam shared/private 解耦、prototype alignment、history residual 和 scene-private calibration MUST 以跨场景可迁移/场景私有表征为核心定义，不得把 teacher-student distillation 作为这些分支生效的必要条件。

#### Scenario: shared/private diagnostics 无 teacher 仍完整
- **WHEN** HiST-Beam shared/private 或 residual calibration 模型在 no-KD 配置下 forward
- **THEN** 模型 MUST 输出该变体要求的 shared/private/residual/prototype/calibration diagnostics
- **AND** diagnostics MUST 不依赖 teacher features 或 teacher logits

#### Scenario: prototype alignment 不使用 teacher soft target
- **WHEN** adapter+prototype 或 path/radio prototype adaptation 计算 prototype loss
- **THEN** prototype target MUST 来自 source prototype、target representation、path/radio semantic assignment 或配置定义的物理/几何 proxy
- **AND** 系统 MUST 不把 teacher prediction distribution 作为默认 prototype target
