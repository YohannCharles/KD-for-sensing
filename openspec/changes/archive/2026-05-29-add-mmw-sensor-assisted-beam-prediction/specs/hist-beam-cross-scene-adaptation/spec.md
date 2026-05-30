## ADDED Requirements

### Requirement: MMW sensor-assisted HiST-Beam profile
HiST-Beam LOSO workflow MUST support an MMW sensor-assisted profile that uses `image`、`gps`、`lidar` 和 `radar` as model inputs. This profile MUST remain separate from existing MMW `image+gps+mmwave` experiments and MUST expose modality profile metadata in plan、run 和 summary artifacts.

#### Scenario: 构建 sensor-assisted 模型配置
- **WHEN** 用户加载 MMW sensor-assisted HiST-Beam 配置
- **THEN** model modalities MUST resolve to `image`、`gps`、`lidar` 和 `radar`
- **AND** student model field defaults MUST include compatible image、gps、lidar 和 radar encoder settings
- **AND** model construction MUST fail with an actionable error if any enabled modality has no compatible sample key

#### Scenario: 变体矩阵沿用 HiST-Beam baseline
- **WHEN** sensor-assisted LOSO plan 生成
- **THEN** plan MUST support source-only、adapter-only、coarse prototype、radio prototype、path prototype、path condition off 和 full fine-tuning baseline variants where available
- **AND** run metadata MUST distinguish sensor-assisted modality profile from `image+gps+mmwave` profile

#### Scenario: summary 输出负迁移诊断
- **WHEN** sensor-assisted quick validation 写出 `loso_summary`
- **THEN** summary MUST include adapted-source Top-K deltas for adaptation variants
- **AND** summary MUST include negative-transfer flags when adapted Top-1 is lower than corresponding source-only Top-1
- **AND** summary MUST preserve trainable ratio and adaptation time fields for parameter-efficiency comparison

#### Scenario: last-beam baseline 不改变输入语义
- **WHEN** evaluation computes last-beam diagnostic baseline
- **THEN** HiST-Beam summary MAY report last-beam Top-K
- **AND** model input construction MUST NOT add previous beam labels or beam power to sensor-assisted sensing modalities because of that diagnostic
