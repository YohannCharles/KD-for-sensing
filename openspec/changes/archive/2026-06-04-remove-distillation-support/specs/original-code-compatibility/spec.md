## ADDED Requirements

### Requirement: 原始兼容不保留 KD 训练模式
项目对原始代码的兼容 MUST 限定于当前保留的数据、模型、训练、评估和指标语义。原始 teacher-student KD 训练模式、旧 argparse KD 参数和旧 KD 配置路径 MUST 不作为兼容目标。

#### Scenario: 旧 KD argparse 参数不兼容
- **WHEN** 用户尝试通过旧参数或 override 启用 `kd_mode`、temperature、alpha、logits KD 或 RKD
- **THEN** 系统 MUST 拒绝该参数
- **AND** 错误信息 MUST 指向当前 supervised/adaptation 配置

