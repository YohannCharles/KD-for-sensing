## REMOVED Requirements

### Requirement: s008 诊断实验矩阵
**Reason**: Raymobtime s008 模态失衡诊断研究线已放弃。
**Migration**: 保留 Raymobtime s008 普通训练、评估和推荐实验配置；不再维护失衡诊断矩阵。

#### Scenario: s008 诊断矩阵退役
- **WHEN** 用户查找模态失衡诊断矩阵
- **THEN** 系统不再要求提供 `configs/raymobtime/s008_modality_imbalance_diagnosis.yaml`

### Requirement: 模态失衡内部诊断产物
**Reason**: gate/drop/gradient/LOS bucket 机制证据用于已退役失衡判定。
**Migration**: 普通 Raymobtime 评估仍输出 objective metrics；机制诊断需另行提出。

#### Scenario: 内部诊断产物退役
- **WHEN** Raymobtime s008 评估完成
- **THEN** 系统不再要求生成模态失衡内部诊断表

### Requirement: 模态失衡判定标准
**Reason**: `confirmed_imbalance` 等结论标签不再代表项目目标。
**Migration**: 不保留旧判定标准。

#### Scenario: 判定标准退役
- **WHEN** 用户分析 Raymobtime s008 run
- **THEN** 系统不再要求输出模态失衡结论标签

### Requirement: s009 外部验证门槛
**Reason**: s009 外部验证门槛服务于 s008 失衡确认流程，已退役。
**Migration**: 若后续引入 s009，应作为独立 dataset 或 experiment change 定义。

#### Scenario: s009 门槛退役
- **WHEN** s008 实验完成
- **THEN** 系统不再要求根据失衡结论决定是否启动 s009

### Requirement: 诊断报告与产物边界
**Reason**: 模态失衡诊断报告已退役。
**Migration**: 本地产物边界继续由通用产物边界要求约束。

#### Scenario: 诊断报告退役
- **WHEN** 用户运行 Raymobtime s008 普通实验
- **THEN** 系统不再要求生成 `diagnosis_report.md` 或失衡 summary JSON
